import os as os
import glob as glob
import cv2 as cv2
import numpy as np

# Initialize variables.
alter_count = 1
keep_count = 1
block_dim = 64

# Initialize path prefixes.
source_prefix = './images/' + str(block_dim) + '/'
training_prefix = './training/' + str(block_dim) + '/'

# Initialize path variables.
blocks_path = source_prefix + 'blocks/'
validation_path = training_prefix + 'validation/'
altered_path = training_prefix + 'altered/'
testing_path = training_prefix + 'testset/'


# Setup main loop to process all frameblocks.
if not os.path.exists(blocks_path):
    print('No inputs found, exiting program.')
    exit()
blocks = os.listdir(blocks_path)
blocks.sort()

# Delete previously output altered frameblocks.
if not os.path.exists(training_prefix):
    os.mkdir(training_prefix)

if os.path.exists(validation_path):
    filelist = glob.glob(validation_path + '*')
    for file in filelist:
        os.remove(file)
else:
    os.mkdir(validation_path)

if os.path.exists(altered_path):
    filelist = glob.glob(altered_path + '*')
    for file in filelist:
        os.remove(file)
else:
    os.mkdir(altered_path)

if os.path.exists(testing_path):
    filelist = glob.glob(testing_path + '*')
    for file in filelist:
        os.remove(file)
else:
    os.mkdir(testing_path)

# Process each block.
for block_index in range(0, len(blocks)):
    block = blocks[block_index]

    img_in_str = blocks_path + block
    print(block)

    # Decide if image will be altered or kept.
    flip = np.random.uniform(0, 1)
    if flip < 0.75:
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
