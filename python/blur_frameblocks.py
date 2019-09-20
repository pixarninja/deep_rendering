import os as os
import glob as glob
import cv2 as cv2
import numpy as np

path_prefix = './images/'
blocks_path = path_prefix + 'blocks/pairs/'
validation_path = './training/validation/'
blurred_path = './training/blurred/'
testing_path = './training/testset/'
blur_count = 1
keep_count = 1

# Delete previously output blurred frameblocks.
if os.path.exists(validation_path):
    filelist = glob.glob(validation_path + '*')
    for file in filelist:
        os.remove(file)
else:
    os.mkdir(validation_path)

if os.path.exists(blurred_path):
    filelist = glob.glob(blurred_path + '*')
    for file in filelist:
        os.remove(file)
else:
    os.mkdir(blurred_path)

if os.path.exists(testing_path):
    filelist = glob.glob(testing_path + '*')
    for file in filelist:
        os.remove(file)
else:
    os.mkdir(testing_path)

# Setup main loop to process all frameblocks.
blocks = os.listdir(blocks_path)
blocks.sort()

# Process each block.
for block_index in range(0, len(blocks)):
    block = blocks[block_index]

    # Store all frames for that block.
    frames = os.listdir(blocks_path + block)
    frames.sort()

    # Process each frame.
    for frame_index in range(0, len(frames)):
        frame = frames[frame_index]
        img_in_str = blocks_path + block + '/' + frame + '/end.jpg'

        # Decide if image will be blurred or kept.
        flip = np.random.uniform(0, 1)
        if flip < 0.75:
            # Store end.jpg image to be blurred.
            img = cv2.cvtColor(cv2.imread(img_in_str), cv2.COLOR_BGR2RGB)

            # Save original end.jpg image for validation.
            img_org_str = validation_path + block + frame + '.jpg'
            cv2.imwrite(img_org_str, img)

            # Blur image.
            img = cv2.GaussianBlur(img, (7, 7), 0)
            
            # Output blurred image.
            img_out_str = blurred_path + block + frame + '.jpg'
            cv2.imwrite(img_out_str, img)
            print(str(blur_count) + ': 0 Blur ' + img_out_str)
            blur_count += 1

        else:
            # Copy end.jpg image to testset.
            img = cv2.cvtColor(cv2.imread(img_in_str), cv2.COLOR_BGR2RGB)
            img_out_str = testing_path + block + frame + '.jpg'
            cv2.imwrite(img_out_str, img)
            print(str(keep_count) + ': 1 Kept ' + img_out_str)
            keep_count += 1
