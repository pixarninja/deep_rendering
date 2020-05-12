import os as os
import cv2 as cv2
import numpy as np

home_dir = './images/'
blocks_dir = 'blocks/pairs/'
orig_dir = './training/validation/'
blur_dir = './training/blurred/'
keep_dir = './training/testset/'
blur_count = 1
keep_count = 1

# Delete previously output blurred frameblocks.
os.system('rm -rf %s' % orig_dir + '*')
os.system('rm -rf %s' % blur_dir + '*')
os.system('rm -rf %s' % keep_dir + '*')

# Setup main loop to process all frameblocks.
blocks = os.listdir(home_dir + blocks_dir)
blocks.sort()

# Process each block.
for block_index in range(0, len(blocks)):
    block = blocks[block_index]

    # Store all frames for that block.
    frames = os.listdir(home_dir + blocks_dir + block)
    frames.sort()

    # Process each frame.
    for frame_index in range(0, len(frames)):
        frame = frames[frame_index]
        img_in_str = home_dir + blocks_dir + block + '/' + frame + '/end.jpg'

        # Decide if image will be blurred or kept.
        flip = np.random.uniform(0, 1)
        if flip < 0.75:
            # Store end.jpg image to be blurred.
            img = cv2.cvtColor(cv2.imread(img_in_str), cv2.COLOR_BGR2RGB)

            # Skip image if entirely black.
            if np.sum(img) == 0:
                continue

            # Save original end.jpg image for validation.
            img_org_str = orig_dir + block + frame + '.jpg'
            cv2.imwrite(img_org_str, img)

            # Blur image.
            img = cv2.GaussianBlur(img, (7, 7), 0)
            
            # Output blurred image.
            img_out_str = blur_dir + block + frame + '.jpg'
            cv2.imwrite(img_out_str, img)
            print(str(blur_count) + ': 0 Blur ' + img_out_str)
            blur_count += 1

        else:
            # Skip image if entirely black.
            img = cv2.cvtColor(cv2.imread(img_in_str), cv2.COLOR_BGR2RGB)
            if np.sum(img) == 0:
                continue

            # Copy end.jpg image to testset.
            img_out_str = keep_dir + block + frame + '.jpg'
            cv2.imwrite(img_out_str, img)
            print(str(keep_count) + ': 1 Kept ' + img_out_str)
            keep_count += 1
