import argparse as argparse
from blocksparse import BlocksparseTransformer
import cv2 as cv2
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import os as os
import tensorflow as tf
import sys as sys
import utils as utils
from utils import shape_list, recomputable

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from keras.datasets import cifar10

parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='size of block to use')
parser.add_argument('--useRange', action='store_true', help='whether or not to use a random range')
parser.add_argument('--alpha', type=float, default=0.75, help='noise constant to use')
parser.add_argument('--beta', type=int, default=7, help='blur constant to use')
parser.add_argument('--alphaPair', type=float, default=0.9, help='noise constant range to use')
parser.add_argument('--betaPair', type=int, default=3, help='blur constant range to use')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--outf', default='./output/', help='folder to output images and model checkpoints')
parser.add_argument('--inType', default='frame', help='input type, one of the following: [frame, cifar]')

opt = parser.parse_args()
print(opt)

pair = None
tag = '%d-%d-%d' % (opt.blockDim, int(opt.alpha * 100), opt.beta)
if opt.useRange:
    opt.alpha = 0.75
    opt.beta = 7
    pair = [opt.alphaPair, opt.betaPair]
    tag = 'range'
outf_path = ('%s%s_outputx%s/' % (opt.outf, opt.inType, tag))

utils.clear_dir(outf_path)

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)
        b = tf.matrix_band_part(tf.ones([n, n]), ctx, 0)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])
        y = tf.transpose(x)
        z = tf.zeros([n, n], dtype=tf.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = tf.equal(tf.floormod(q - k, stride), 0)
        c3 = tf.logical_and(c1, c2)
        b = tf.cast(c3, tf.float32)
    else:
        raise ValueError('Not yet implemented')
    b = tf.reshape(b, [1, 1, n, n])
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, '{bT_ctx}, {blocksize}'.format(bT_ctx=bT_ctx, blocksize=blocksize)
    n, t, embd = shape_list(x)
    x = tf.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [n, t, embd])
    return x


def split_heads(x, n):
    return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


@recomputable('attention_impl')
def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = shape_list(k)[2]
    mask = tf.to_float(get_attn_mask(n_timesteps, attn_mode, local_attn_ctx))
    w = tf.matmul(q, k, transpose_b=True)
    scale_amount = 1.0 / np.sqrt(shape_list(q)[-1])
    orig_dtype = q.dtype
    if orig_dtype == tf.float16:
        w = tf.cast(w, tf.float32)
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = tf.nn.softmax(w)
    w = tf.cast(w, orig_dtype)
    a = tf.matmul(w, v)
    a = merge_heads(a)
    return a


@recomputable('blocksparse_attention_impl')
def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None,
                               blocksize=32, num_verts=None, vertsize=None):
    n_ctx = shape_list(q)[1]
    if attn_mode == 'strided':
        # Strided attention is implemented on the transposed matrix to provide greater block sparsity
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = shape_list(q)[-1] // heads
    bst = get_blocksparse_obj(n_ctx, heads, attn_mode, blocksize, local_attn_ctx, num_verts, vertsize)
    scale_amount = tf.cast(1.0 / np.sqrt(n_state), tf.float32)
    w = bst.query_key_op(q, k)
    w = bst.masked_softmax(w, scale=scale_amount)
    a = bst.weight_value_op(w, v)
    if attn_mode == 'strided':
        n, t, embd = shape_list(a)
        bT_ctx = n_ctx // local_attn_ctx
        a = tf.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = tf.transpose(a, [0, 2, 1, 3])
        a = tf.reshape(a, [n, t, embd])
    return a


def get_blocksparse_obj(n_ctx, n_heads, attn_mode, blocksize=32, local_attn_ctx=None, num_verts=4, vertsize=1):
    '''Defines the block-level sparsity pattern in the attention matrix. Enabled blocks
    will have the callback called on them in order to define a positionwise sparsity mask.'''
    n_bctx = n_ctx // blocksize
    layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
    extra_diagonals = None
    block_chunks = None

    if attn_mode in ['all', 'fixed']:
        pass
    elif attn_mode == 'local':
        assert local_attn_ctx % blocksize == 0
        extra_diagonals = local_attn_ctx // blocksize
    elif attn_mode == 'strided':
        bT_ctx = n_ctx // local_attn_ctx
        assert bT_ctx % blocksize == 0
        block_chunks = bT_ctx // blocksize
    else:
        raise ValueError('attn mode {attn_mode} invalid'.format(attn_mode=attn_mode))

    if attn_mode == 'fixed':
        assert n_heads % num_verts == 0
        lctx = local_attn_ctx
        stride = lctx // blocksize
        assert vertsize <= stride
        assert stride % vertsize == 0
        indices = [i for i in range(stride - 1, -1, -1)]
        indices = np.array(indices).reshape([-1, vertsize])
        if num_verts == 1:
            layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
            for idx in indices[0]:
                layout[:, idx::stride] = 1
            for q_idx in range(n_bctx):
                # Each thing can attend to its local block
                row = q_idx // stride
                layout[q_idx, row * stride:(row + 1) * stride] = 1
                # Any query cannot attend to keys above it
                layout[q_idx, q_idx + 1:] = 0
        else:
            layouts = []
            indices = indices[:num_verts]
            for h in range(n_heads):
                layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
                subindices = indices[h % num_verts]
                for idx in subindices:
                    layout[:, idx::stride] = 1
                for q_idx in range(n_bctx):
                    # Each position can attend to its local block
                    row = q_idx // stride
                    layout[q_idx, row * stride:(row + 1) * stride] = 1
                    # Any query cannot attend to keys above it
                    layout[q_idx, q_idx + 1:] = 0
                layouts.append(layout)
            layout = np.array(layouts)
    else:
        for q_idx, k_idx in np.ndindex(n_bctx, n_bctx):
            if k_idx > q_idx:
                layout[q_idx, k_idx] = 0
            if extra_diagonals and k_idx + extra_diagonals < q_idx:
                layout[q_idx, k_idx] = 0
            if block_chunks is not None:
                layout[q_idx, k_idx] = 0
                offset = q_idx % block_chunks
                if k_idx + offset >= q_idx and k_idx <= q_idx:
                    layout[q_idx, k_idx] = 1
    bst = BlocksparseTransformer(layout, block_size=blocksize,
                                 mask_callback=get_callback(attn_mode, local_attn_ctx),
                                 heads=n_heads)
    return bst


def get_callback(attn_mode, local_attn_ctx=None):
    '''Defines a function which returns the positionwise sparsity pattern for every block
    that is enabled in the blocksparse object
    '''
    def cb(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.ones(blk_shape, dtype=np.bool)

        # on the diagonal blocks mask out the upper diagonal
        if qry_idx == key_idx:
            for q, k in np.ndindex(blk_shape):
                if k > q:
                    mask[q, k] = 0
        if attn_mode in ['all', 'strided', 'fixed']:
            return mask
        if attn_mode == 'local':
            bandwidth = local_attn_ctx
            # convert group indices to absolute indices and mask
            # according to that
            q_pos = blk_shape[0] * qry_idx
            k_pos = blk_shape[1] * key_idx
            for q, k in np.ndindex(blk_shape):
                q_ = q + q_pos
                k_ = k + k_pos
                if k_ > q_ or k_ + bandwidth <= q_:
                    mask[q, k] = 0
            return mask
        raise ValueError
    return cb

def get_data(dataloader):
        
    images, labels = next(train_data_gen)
    high_res, low_res = process_images(images)
    print(high_res.shape)
    show_batch(high_res, opt.outf + 'high_test.png')
    show_batch(low_res, opt.outf + 'low_test.png')

    return high_res, low_res, train_data_gen

def process_images(data):
    high_res, _ = data

    high_res = normalize_images(high_res)
    low_res = []
    for i, img in enumerate(high_res):
        alt = utils.alter_image(img.transpose(1, 2, 0), opt.alpha, opt.beta)
        low_res.append(alt.transpose(2, 0, 1))

    return high_res, normalize_images(low_res)

def normalize_images(images):
    return (np.array(images) - np.array(images).min(0)) / np.array(images).ptp(0)

def to_tensor(src, n_batch, n_ctx, n_embd):
    return tf.convert_to_tensor(np.asarray(src).reshape(n_batch, n_ctx, n_embd))

def show_batch(image_batch, path):
    max_index = 25

    if len(image_batch) < max_index:
        plt.imshow(image_batch[0].astype('float32'))
        plt.axis('off')
    else:
        plt.figure(figsize=(10,10))
        for n in range(max_index):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n].astype('float32'))
            plt.title(str(n))
            plt.axis('off')

    plt.savefig(path)
    plt.close('all')

def save_img(image, path):
    data = torch.from_numpy(image)
    vutils.save_image(data, path, normalize=True)

if __name__ == '__main__':
# Load data.
    if opt.inType == 'frame':
        transform = transforms.Compose([transforms.RandomCrop(opt.blockDim), transforms.ToTensor()])
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        data_prefix = '/app/training/' + str(opt.blockDim) + '/'
        dataset = datasets.ImageFolder(root=data_prefix + 'validation/', transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)
        testset = datasets.ImageFolder(root=data_prefix + 'testset/', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False)

        # Generate training data.
        x_train = []
        y_train = []
        for i, data in enumerate(dataloader, 0):
            high_res_real = data[0]
            if np.shape(high_res_real)[0] != opt.batchSize:
                continue

            # Downsample images to low resolution.
            for j in range(opt.batchSize):
                x_train.append(utils.alter_image(high_res_real[j].numpy().transpose(1, 2, 0), opt.alpha, opt.beta, pair=pair))
                y_train.append(normalize(high_res_real[j]))

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)

        # Generate testing data.
        x_test = []
        y_test = []
        for i, data in enumerate(testloader, 0):
            high_res_real = data[0]
            if np.shape(high_res_real)[0] != opt.batchSize:
                continue

            # Downsample images to low resolution.
            for j in range(opt.batchSize):
                x_test.append(utils.alter_image(high_res_real[j].numpy().transpose(1, 2, 0), opt.alpha, opt.beta, pair=pair))
                y_test.append(normalize(high_res_real[j]))

        x_test = torch.stack(x_test)
        y_test = torch.stack(y_test)

    elif opt.inType == 'cifar':
        (train_data, _), (test_data, _) = cifar10.load_data()

        # Generate training data.
        x_train = []
        y_train = []
        for i, data in enumerate(train_data, 0):
            data = data / 255.

            # Downsample image to low resolution.
            x_train.append(utils.alter_image(np.array(data), opt.alpha, opt.beta, pair=pair))
            y_train.append(torch.from_numpy(data.transpose(2, 0, 1)))

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)

        x_test = []
        y_test = []
        for i, data in enumerate(test_data, 0):
            data = data / 255.

            # Downsample image to low resolution.
            x_test.append(utils.alter_image(np.array(data), opt.alpha, opt.beta, pair=pair))
            y_test.append(torch.from_numpy(data.transpose(2, 0, 1)))

        x_test = torch.stack(x_test)
        y_test = torch.stack(y_test)

    else:
        print('ERROR: Input data type not recognized')
        exit(1)

    # Plot 25 sample images.
    plt.figure(figsize=(10,10))
    for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(np.asarray(x_train[i]).transpose(1, 2, 0))
            #plt.title(str(i))
            plt.axis('off')
    plt.savefig('{}/{}_samplesx{}.png'.format(opt.outf, opt.inType, tag))
    plt.close('all')

    n_samples = int(x_train.shape[0] / opt.batchSize)
    number = -1
    print('\nBatch Size: {}, Batches: {}'.format(opt.batchSize, n_samples))

    # Set up application.
    utils.make_dir(opt.outf)
    utils.make_dir(outf_path)

    # Number of samples/batches? (4)
    # Number of time steps (batch_size / n_batch) --> number of pixels?
    # Number of features (256 x 3)
    dtype = tf.float32
    n_batch = 4
    n_ctx = opt.blockDim**2
    n_embd = int(opt.blockDim**2 * 3 / 4)
    blocksize = 32
    
    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph.
    with tf.Session() as sess:
        sess.run(init)
     
        # Training cycle.
        for epoch in range(opt.nEpochs):
            avg_cost = 0.0
            total_samples = 0

            for i in range(n_samples):
                low_res = np.float32(x_train[i * opt.batchSize:(i + 1) * opt.batchSize].numpy())
                high_res = np.float32(y_train[i * opt.batchSize:(i + 1) * opt.batchSize].numpy())

                if high_res.shape[0] < opt.batchSize or low_res.shape[0] < opt.batchSize:
                    continue

                total_samples += n_samples
                low_res_d = low_res.transpose(0, 2, 3, 1).reshape(n_batch, n_ctx, n_embd)
                high_res_d = high_res.transpose(0, 2, 3, 1).reshape(n_batch, n_ctx, n_embd)
                q = to_tensor(low_res_d, n_batch, n_ctx, n_embd)
                k = to_tensor(low_res_d, n_batch, n_ctx, n_embd)
                v = to_tensor(high_res_d, n_batch, n_ctx, n_embd)

                # first step of strided attention.
                local_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=blocksize, blocksize=blocksize, recompute=True)

                # Second step of strided attention.
                strided_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=blocksize, blocksize=blocksize, recompute=True)

                # Run model and get new losses.
                cost = tf.reduce_mean(tf.square(strided_attn_bs - high_res.reshape(n_batch, n_ctx, n_embd)))
                cost_val, strided_bs = sess.run([cost, strided_attn_bs], feed_dict={q: q.eval(), k: k.eval(), v: v.eval()})
                avg_cost += cost_val

                # Log every 10 batches of images processed.
                iteration = x_train.shape[0] * epoch + i * opt.batchSize
                print('Iteration: {}, Cost: {}'.format(iteration, cost_val))
                number += 1
                save_img(high_res[0], '%sreal_%03d.png' % (outf_path, number))
                save_img(low_res[0], '%salt_%03d.png' % (outf_path, number))
                save_img(normalize_images(strided_bs.reshape(opt.batchSize, opt.blockDim, opt.blockDim, 3))[0].transpose(2, 0, 1), '%sfake_%03d.png' % (outf_path, number))

            print('Average Cost: {}'.format(avg_cost / total_samples))

        print(strided_bs[0])

        # Generate results on test data for evaluation.
        n_samples = int(x_test.shape[0] / opt.batchSize)
        number = -1
        print('\nBatch Size: {}, Batches: {}'.format(opt.batchSize, n_samples))

        altr_path = outf_path + 'low_res/'
        real_path = outf_path + 'high_res_real/'
        fake_path = outf_path + 'high_res_fake/'
        utils.clear_dir(altr_path)
        utils.clear_dir(real_path)
        utils.clear_dir(fake_path)

        for i in range(n_samples):
            low_res = np.float32(x_test[i * opt.batchSize:(i + 1) * opt.batchSize].numpy())
            high_res = np.float32(y_test[i * opt.batchSize:(i + 1) * opt.batchSize].numpy())

            if high_res.shape[0] < opt.batchSize or low_res.shape[0] < opt.batchSize:
                continue

            low_res_d = low_res.transpose(0, 2, 3, 1).reshape(n_batch, n_ctx, n_embd)
            high_res_d = high_res.transpose(0, 2, 3, 1).reshape(n_batch, n_ctx, n_embd)
            q = to_tensor(low_res_d, n_batch, n_ctx, n_embd)
            k = to_tensor(low_res_d, n_batch, n_ctx, n_embd)
            v = to_tensor(high_res_d, n_batch, n_ctx, n_embd)

            # first step of strided attention.
            local_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=blocksize, blocksize=blocksize, recompute=True)

            # Second step of strided attention.
            strided_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=blocksize, blocksize=blocksize, recompute=True)

            # Run model and get new losses.
            cost = tf.reduce_mean(tf.square(strided_attn_bs - high_res.reshape(n_batch, n_ctx, n_embd)))
            cost_val, strided_bs = sess.run([cost, strided_attn_bs], feed_dict={q: q.eval(), k: k.eval(), v: v.eval()})

            # Save all images.
            iteration = x_test.shape[0] * epoch + i * opt.batchSize
            print('Iteration: {}'.format(iteration))
            for j in range(high_res.shape[0]):
                number += 1
                save_img(high_res[j], '%s%d.png' % (real_path, number))
                save_img(low_res[j], '%s%d.png' % (altr_path, number))
                save_img((strided_bs.reshape(opt.batchSize, opt.blockDim, opt.blockDim, 3))[j].transpose(2, 0, 1), '%s%d.png' % (fake_path, number))
