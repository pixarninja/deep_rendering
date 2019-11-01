<<<<<<< HEAD
import os as os
import glob as glob
import cv2 as cv2
import numpy as np

# Initialize seed variables.
block_dim = 32
block_offset = 1
start_and_end = False

# Initialize path prefixes.
path_prefix = './images/'
src_prefix = '../source/animations/'
buffer_prefix = path_prefix + 'buffer/'

# Initialize path variables.
frames_path = src_prefix + 'glass_full/'
training_path = path_prefix + 'blocks/'
=======
import argparse
import cv2 as cv2
import glob as glob
import numpy as np
import os as os
import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='dimension of frameblocks')
parser.add_argument('--blockOffset', type=int, default=1, help='offset for blocks, > 1 blocks will overlap')
parser.add_argument('--startBlock', type=bool, default=False, help='whether to record the start block as well')

opt = parser.parse_args()
print(opt)

# Initialize seed variables.
block_dim = opt.blockDim
block_offset = opt.blockOffset
start_and_end = opt.startBlock

# Initialize path prefixes.
path_prefix = './images/' + str(block_dim) + '/'
training_prefix = './training/' + str(block_dim) + '/'
src_prefix = '../source/animations/'
buffer_prefix = path_prefix

# Initialize path variables.
frames_path = src_prefix + 'glass_full/'
training_path = training_prefix + 'blocks/'
>>>>>>> b01d3b46eddf6099278773795edd739282c4ff6e
shadow_img_path = path_prefix + 'shadow/'
roi_img_path = path_prefix + 'roi/'
shadow_buff_path = buffer_prefix + 'shadows/'
frame_buff_path = buffer_prefix + 'frames/'

# Delete previously output frameblocks, and buffer shadows and buffer frames.
<<<<<<< HEAD
if os.path.exists(training_path):
    filelist = glob.glob(training_path + '*')
    for file in filelist:
        os.remove(file)
else:
    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)
    os.mkdir(training_path)

if not os.path.exists(buffer_prefix):
    os.mkdir(buffer_prefix)
    os.mkdir(shadow_buff_path)
    os.mkdir(frame_buff_path)
else:
    if os.path.exists(shadow_buff_path):
        filelist = glob.glob(shadow_buff_path + '*')
        for file in filelist:
            os.remove(file)
    else:
        os.mkdir(shadow_buff_path)

    if os.path.exists(frame_buff_path):
        filelist = glob.glob(frame_buff_path + '*')
        for file in filelist:
            os.remove(file)
    else:
        os.mkdir(frame_buff_path)
=======
utils.make_dir(training_prefix)
utils.clear_dir(training_path)

utils.make_dir(buffer_prefix)
utils.clear_dir(shadow_buff_path)
utils.clear_dir(frame_buff_path)
>>>>>>> b01d3b46eddf6099278773795edd739282c4ff6e

# Setup main loop to process all frames in an animation.
frames = os.listdir(frames_path)
frames.sort()

# Process each frame.
for frame_index in range(0, len(frames), 2):
    frame = frames[frame_index]
    frame_str_out = 'frame' + str(frame_index + 1)
    block_index = 1
    block_str_out = 'block' + str(block_index + 1)

    # If the frame index is 0, store all frameblocks.
    if frame_index < 1 and start_and_end:
        # Initialize seed variables.
        img_str_1 = frames_path + frames[frame_index]

        # Choose smallest boundaries.
        img_1 = cv2.imread(img_str_1)
        #img_1 = cv2.resize(img_1, (0,0), fx=0.5, fy=0.5) 
        height, width = img_1.shape[:2]

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
                print(str(block_index) + ". Frameblock: (" + str(left) + ", " + str(top) + "), (" + str(right) + ", " + str(bottom) + "))")

                # Store window contents as image.
                img_str_out = training_path + frame_str_out + block_str_out
                img_roi = img_1[top:bottom, left:right]
                if start_and_end:
                    suffix = '_end.jpg'
                else:
                    suffix = '.jpg'
                cv2.imwrite(img_str_out + suffix, img_roi)

                # Increase frameblock index.
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

    # Otherwise process as normal.
    else:
        # Initialize seed variables.
        img_str_1 = frames_path + frames[frame_index - 2]
        img_str_2 = frames_path + frames[frame_index]
        img_str_shd = shadow_img_path + 'frame' + str(frame_index) + '.jpg'
        img_str_roi = roi_img_path + 'frame' + str(frame_index) + '.jpg'

        img_1 = cv2.imread(img_str_1)
        #img_1 = cv2.resize(img_1, (0,0), fx=0.5, fy=0.5) 
        height_1, width_1 = img_1.shape[:2]

        img_2 = cv2.imread(img_str_2)
        #img_2 = cv2.resize(img_2, (0,0), fx=0.5, fy=0.5) 
        height_2, width_2 = img_2.shape[:2]

        # Choose smallest boundaries.
        height = height_1
        width = width_1
        if height_1 > height_2:
            height = height_2
        if width_1 > width_2:
            width = width_2

        img_out = np.ones((height, width, 3), np.uint8)

        # Calculate XOR image and pixel sum.
        print("Processing pixels of images, \'" + img_str_1 + "\' and \'" + img_str_2 + "\'")
        img_xor = cv2.bitwise_xor(img_1, img_2)
        pixel_sum = np.sum(img_xor)
        img_out = cv2.bitwise_not(cv2.cvtColor(img_xor, cv2.COLOR_BGR2GRAY))

        # Continue to next frame if no changes were found.
        if pixel_sum <= 255:
            print("No major changes found, continuing to next image.")
            continue

        # Write image.
        cv2.imwrite(img_str_shd, img_out)
        print("Wrote shadow image, \'" + img_str_shd + "\'")

        # Calculate the pixel_ratio.
        print("Total pixel sum: " + str(pixel_sum))
        pixel_ratio = pixel_sum * 1.0 / (255 * width * height)
        print("Pixel ratio: " + str(pixel_ratio))

        # Create a clone of input image and draw ROIs on top of it.
        img_roi_all = cv2.imread(img_str_shd)

        # Create sliding window.
        left = 0
        right = block_dim
        top = 0
        bottom = block_dim
        pixel_sum = 0
        cap = np.power(block_dim, 2) * 255 * pixel_ratio
        print("Cap found: " + str(cap))

        # Find the Region Of Interest (ROI).
        while bottom <= height:
            if bottom == height:
                bottom -= 1
            while right <= width:
                if right == width:
                    right -= 1
                found_x = False
                dirty = False
                pixel_sum = 0
                img_buff_str = shadow_buff_path + 'block' + str(block_index) + '.jpg'
                img_buff = cv2.imread(img_buff_str)
                if img_buff is None:
                    img_buff = np.zeros((block_dim, block_dim, 3), np.uint8)
                
                # ROI pixel processing
                for y in range(top, bottom + 1):
                    for x in range(left, right + 1):

                        # Store buffer pixel and calculate pixel_sum.
                        img_buff[y - top - 1, x - left - 1] += 255 - img_out[y, x]

                        if img_buff[y - top - 1, x - left - 1][0] > 0:
                            dirty = True
                            if img_buff[y - top - 1, x - left - 1][0] > 255:
                                img_buff[y - top - 1, x - left - 1] = 255
                        pixel_sum += img_buff[y - top - 1, x - left - 1][0]
                        
                        # Test if the cap was met.
                        if pixel_sum >= cap:

                            # Draw ROI on clone image.
                            cv2.rectangle(img_roi_all, (left + 1, top + 1), (right - 1, bottom - 1), (255, 0, 0), 1)
                            cv2.putText(img_roi_all, str(block_index), (left + 3, bottom - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 0), 1, 1)
                            print(str(block_index) + ". Sum: " + str(pixel_sum) + ", Frameblock: (" + str(left) + ", " + str(top) + "), (" + str(right) + ", " + str(bottom) + "))")
                            
                            # Store window contents as image.
                            img_str_out = training_path + frame_str_out + block_str_out
                            img_roi = img_2[top:bottom, left:right]
                            if start_and_end:
                                suffix = '_end.jpg'
                            else:
                                suffix = '.jpg'
                            cv2.imwrite(img_str_out + suffix, img_roi)
                            
                            # Exit both for loops.
                            found_x = True
                            break
                    if found_x:
                        break

                # If frameblock was used delete buffer shadow.
                if found_x:
                    # If a buffered shadow image was used, delete it.
                    if os.path.exists(img_buff_str):
                        os.remove(img_buff_str)

                    # If there is a buffered frame for the block use it as the starting frame.
                    img_buff_str = frame_buff_path + 'block' + str(block_index) + '.jpg'
                    if os.path.exists(img_buff_str):
                        # Move and rename file.
                        if start_and_end:
                            suffix = '_start.jpg'
                            os.rename(img_buff_str, img_str_out + suffix)

                    # Otherwise export the ROI of the first image as the starting frame.
                    elif start_and_end:
                        # Store window contents as image.
                        img_roi = img_1[top:bottom, left:right]
                        suffix = '_start.jpg'
                        cv2.imwrite(img_str_out + suffix, img_roi)

                # Otherwise export the shadow and frame ROI to be used next time.
                else:
                    # If the shadow image already exists, update it using a linear add.
                    img_buff_str = shadow_buff_path + 'block' + str(block_index) + '.jpg'

                    if os.path.exists(img_buff_str):
                        # Add the values of the current shadow image and previous shadow image.
                        img_prev_buff = cv2.imread(img_buff_str)
                        cv2.addWeighted(img_prev_buff, 1.0, img_buff, 1.0, 0.0, img_buff)

                        # Store updated shadow image (don't update the old frame ROI).
                        cv2.imwrite(img_buff_str, img_buff)
                        print('Updated shadow image, \'block' + str(block_index) + '.jpg\'')

                        #    # Draw Shadow ROI on clone image.
                        #    cv2.rectangle(img_roi_all, (left + 1, top + 1), (right - 1, bottom - 1), (255, 0, 255), 1)
                        #    cv2.putText(img_roi_all, str(block_index), (left + 3, bottom - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 255), 1, 1)

                    # Else if a black pixel was found write a new shadow image.
                    elif dirty:
                        cv2.imwrite(img_buff_str, img_buff)
                        print('Wrote new shadow image, \'block' + str(block_index) + '.jpg\'')

                        # Store frame ROI image.
                        img_buff_str = frame_buff_path + 'block' + str(block_index) + '.jpg'
                        img_roi = img_1[top:bottom, left:right]
                        cv2.imwrite(img_buff_str, img_roi)

                        #    Draw Shadow ROI on clone image.
                        #    cv2.rectangle(img_roi_all, (left + 1, top + 1), (right - 1, bottom - 1), (0, 0, 0), 1)
                        #    cv2.putText(img_roi_all, str(block_index), (left + 3, bottom - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1, 1)

                    # Otherwise notify that the block has been processed with no export.
                    # Commented out for now to save time printing.
                    #else:
                    #    print(str(block_index) + ". No export")

                    #    # Draw Shadow ROI on clone image.
                    #    cv2.rectangle(img_roi_all, (left + 1, top + 1), (right - 1, bottom - 1), (0, 0, 255), 1)
                    #    cv2.putText(img_roi_all, str(block_index), (left + 3, bottom - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, 1)

                # Increase frameblock index.
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

        # Write image.
        cv2.imwrite(img_str_roi, img_roi_all)
