import argparse
import cv2
import os
import numpy as np
import utils

def generate_imageblocks(path):
    # Initialize seed variables.
    block_dim = opt.blockDim
    block_offset = opt.blockOffset

    # Initialize path variables.
    images_path = path + '/images/'
    
    training_prefix = path + '/training/'
    block_prefix = training_prefix + str(block_dim)
    training_path = block_prefix + '/blocks/'

    # Delete previously output imageblocks, and buffer shadows and buffer images.
    utils.make_dir(training_prefix)
    utils.make_dir(block_prefix)
    utils.clear_dir(training_path)

    # Setup main loop to process all images in an animation.
    images = os.listdir(images_path)
    images.sort()

    # Process each image.
    for image_index in range(0, len(images)):
        image = images[image_index]
        image_str_out = 'image' + str(image_index + 1)
        block_index = 1
        block_str_out = 'block' + str(block_index + 1)

        # Initialize seed variables.
        img_str = images_path + images[image_index]

        # Choose smallest boundaries.
        img = cv2.imread(img_str)
        height, width = img.shape[:2]

        # Create sliding window.
        left = 0
        right = block_dim
        top = 0
        bottom = block_dim

        # Find the Region Of Interest (ROI).
        while bottom <= height:
            if bottom == height:
                bottom -= 1
            while right <= width:
                if right == width:
                    right -= 1

                # ROI pixel processing
                print(str(block_index) + ". imageblock: (" + str(left) + ", " + str(top) + "), (" + str(right) + ", " + str(bottom) + "))")

                # Store window contents as image.
                img_str_out = training_path + image_str_out + block_str_out
                img_roi = img[top:bottom, left:right]
                cv2.imwrite(img_str_out + '.jpg', img_roi)
                print('wrote to ' + img_str_out)

                # Increase imageblock index.
                block_index += 1
                block_str_out = 'block' + str(block_index + 1)
                
                # Shift horizontally.
                left += int(block_dim / block_offset)
                right += int(block_dim / block_offset)

            # Shift vertically.
            top += int(block_dim / block_offset)
            bottom += int(block_dim / block_offset)
            left = 0
            right = block_dim
            
parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='dimension of imageblocks')
parser.add_argument('--blockOffset', type=int, default=1, help='offset for blocks, > 1 blocks will overlap')

opt = parser.parse_args()
print(opt)
            
# Generate frameblocks off of parsed images.
base_path = './datasets/VisualGenome'
generate_imageblocks(base_path)