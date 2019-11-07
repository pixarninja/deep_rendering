import argparse
import cv2 as cv2
import glob as glob
import numpy as np
import os as os
import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='dimension of frameblocks')
parser.add_argument('--probability', type=float, default=0.75, help='probability for altering image')

opt = parser.parse_args()
print(opt)

# Initialize variables.
alter_count = 1
keep_count = 1
block_dim = opt.blockDim
prob = opt.probability

# Initialize path prefixes.
training_prefix = './training/' + str(block_dim) + '/'

# Initialize path variables.
blocks_path = training_prefix + 'original/blocks/'
validation_prefix = training_prefix + 'validation/'
validation_path = validation_prefix + 'blocks/'
altered_prefix = training_prefix + 'altered/'
altered_path = altered_prefix + 'blocks/'
testing_prefix = training_prefix + 'testset/'
testing_path = testing_prefix + 'blocks/'


# Setup main loop to process all frameblocks.
if not os.path.exists(blocks_path):
    print('No inputs found, exiting program.')
    exit()
blocks = os.listdir(blocks_path)
blocks.sort()

# Delete previously output altered frameblocks.
utils.make_dir(training_prefix)

utils.make_dir(validation_prefix)
utils.clear_dir(validation_path)

utils.make_dir(altered_prefix)
utils.clear_dir(altered_path)

utils.make_dir(testing_prefix)
utils.clear_dir(testing_path)

# Process each block.
for block_index in range(0, len(blocks)):
    block = blocks[block_index]

    img_in_str = blocks_path + block
    print(block)

    # Decide if image will be altered or kept.
    flip = np.random.uniform(0, 1)
    if flip < prob:
        # Store end.jpg image to be altered.
        img = cv2.imread(img_in_str)

        # Save original end.jpg image for validation.
        img_org_str = validation_path + block
        cv2.imwrite(img_org_str, img)

        # Add noise.
        noise = np.random.normal(loc=0, scale=1, size=img.shape).astype('uint8')
        img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)

        # Gaussian blur.
        img = cv2.GaussianBlur(img, (7, 7), 0)

        # Output altered image.
        img_out_str = altered_path + block
        cv2.imwrite(img_out_str, img)
        print(str(alter_count) + ': 0 Altered ' + img_out_str)
        alter_count += 1

    else:
        # Copy end.jpg image to testset.
        img = cv2.imread(img_in_str)
        img_out_str = testing_path + block
        cv2.imwrite(img_out_str, img)
        print(str(keep_count) + ': 1 Kept ' + img_out_str)
        keep_count += 1
