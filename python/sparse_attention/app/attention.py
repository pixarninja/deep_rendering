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

parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='size of block to use')
parser.add_argument('--alpha', type=float, default=0.75, help='noise constant to use')
parser.add_argument('--beta', type=int, default=7, help='blur constant to use')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--outf', default='./output/', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

def show_batch(image_batch, label_batch, class_names, output_dir):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n]==1][0].title())
        plt.axis('off')

    plt.savefig(output_dir)

if __name__ == '__main__':
    # Find data path.
    data_path = '/app/training/' + str(opt.blockDim) + '/'
    classes = np.array([os.path.basename(item) for item in glob.glob(data_path + '*')])
    print(classes)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = image_generator.flow_from_directory(directory=str(data_path),
                                                     batch_size=opt.batchSize,
                                                     shuffle=True,
                                                     target_size=(opt.imageSize, opt.imageSize),
                                                     classes = list(classes))

    high_res, labels = next(train_data_gen)
    show_batch(high_res, labels, classes, opt.outf + 'high_res.png')

    # Create altered image pairs from the current batch.
    low_res = []
    for i, img in enumerate(high_res):
        alt = utils.alter_image(img, opt.alpha, opt.beta)
        alt_norm = cv2.normalize(alt, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        low_res.append(alt_norm)

    show_batch(low_res, labels, classes, opt.outf + 'low_res.png')

    #dtype = tf.float32
    #n_batch = 4
    #n_ctx = 1024
    #n_embd = 256
    #blocksize = 32

    dtype = tf.float32
    n_batch = 4
    n_ctx = int(opt.batchSize / n_batch)
    n_embd = int(opt.imageSize**2) * 3
    blocksize = 32

    # query, key, values should be batch x time x dim.
    print(len(high_res), high_res[0].shape)
    np.asarray(high_res).reshape(n_batch, n_ctx, n_embd)
    
    q = tf.random_normal(shape=[n_batch, n_ctx, n_embd], dtype=dtype)
    k = tf.random_normal(shape=[n_batch, n_ctx, n_embd], dtype=dtype)
    v = tf.random_normal(shape=[n_batch, n_ctx, n_embd], dtype=dtype)

    # first step of strided attention.
    local_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=32, blocksize=blocksize, recompute=True)
    local_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=32, recompute=True)

    # Second step of strided attention.
    strided_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=32, blocksize=blocksize, recompute=True)
    strided_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=32, recompute=True)

    sess = tf.Session()
    strided_tf, strided_bs = sess.run([strided_attn_tf, strided_attn_bs])

    print(strided_tf[0])
    print(strided_bs[0])
