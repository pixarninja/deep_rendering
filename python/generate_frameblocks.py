import os as os
import cv2 as cv2
import numpy as np

block_dim = 32
block_offset = 1

# Delete previously output frameblocks, and buffer shadows and buffer frames.
os.system('rm -rf %s' % './images/blocks/pairs/*')
os.system('rm -rf %s' % './images/blocks/buffer/shadows/*')
os.system('rm -rf %s' % './images/blocks/buffer/frames/*')

# TODO: Implement inside a loop.

# Initialize seed variables.
img_str_1 = './images/samples/frame_1.jpg'
img_str_2 = './images/samples/frame_3.jpg'
img_str_shd = './images/frame_shadow.jpg'
img_str_roi = './images/frameblock_roi.jpg'

img_1 = cv2.cvtColor(cv2.imread(img_str_1), cv2.COLOR_BGR2RGB)
height_1, width_1 = img_1.shape[:2]
img_2 = cv2.cvtColor(cv2.imread(img_str_2), cv2.COLOR_BGR2RGB)
height_2, width_2 = img_2.shape[:2]

# Choose smallest boundaries.
height = height_1
width = width_1
if height_1 > height_2:
    height = height_2
if width_1 > width_2:
    width = width_2

img_out = np.ones((height, width, 3), np.uint8)
total_pixel_sum = 0

# Loop through each pixel.
for y in range(0, height):
    for x in range(0, width):
        pixel_out = abs(img_1[y, x] ^ img_2[y, x])
        pixel_sum = int(pixel_out[0]) + int(pixel_out[1]) + int(pixel_out[2])
        if pixel_sum > 255:
            bw_val = 255
        else:
            bw_val = np.uint8(pixel_sum)
        total_pixel_sum += bw_val
        bw_val = 255 - bw_val
        img_out[y, x] = bw_val

# Write image.
cv2.imwrite(img_str_shd, img_out)

# Calculate the pixel_ratio.
print("Total pixel sum: " + str(total_pixel_sum))
pixel_ratio = total_pixel_sum * 1.0 / (255 * width * height)
print("Pixel ratio: " + str(pixel_ratio))

# Create a clone of input image and draw ROIs on top of it.
img_roi_all = cv2.imread(img_str_shd)

# Create sliding window.
left = 0
right = block_dim
top = 0
bottom = block_dim
frame_index = 1
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
        img_buff_str = './images/blocks/buffer/shadows/block' + str(frame_index) + '.jpg'
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
                    # Mark as 
                    dirty = True

                    # Draw ROI on clone image.
                    cv2.rectangle(img_roi_all, (left + 1, top + 1), (right - 1, bottom - 1), (255, 0, 0), 1)
                    cv2.putText(img_roi_all, str(frame_index), (left + 3, bottom - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 0, 0), 1, 1)
                    print(str(frame_index) + ". Sum: " + str(pixel_sum) + ", Frameblock: (" + str(left) + ", " + str(top) + "), (" + str(right) + ", " + str(bottom) + "))")
                    
                    # Setup storage.
                    img_out_buff = './images/blocks/pairs/block' + str(frame_index)
                    if not os.path.exists(img_out_buff):
                        os.mkdir(img_out_buff)

                    # Store window contents as image.
                    img_roi = img_2[top:bottom, left:right]
                    cv2.imwrite(img_out_buff + '/end.jpg', img_roi)
                    
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
            img_buff_str = './images/blocks/buffer/frames/block' + str(frame_index) + '.jpg'
            if os.path.exists(img_buff_str):
                # Setup storage.
                img_out_str = './images/blocks/pairs/block' + str(frame_index)
                if not os.path.exists(img_out_str):
                    os.mkdir(img_out_str)

                # Move and rename file.
                os.rename(img_buff_str, img_out_str + '/start.jpg')

            # Otherwise export the ROI of the first image as the starting frame.
            else:
                # Setup storage.
                img_out_str = './images/blocks/pairs/block' + str(frame_index)
                if not os.path.exists(img_out_str):
                    os.mkdir(img_out_str)

                # Store window contents as image.
                img_roi = img_1[top:bottom, left:right]
                cv2.imwrite(img_out_str + '/start.jpg', img_roi)

        # Otherwise export the shadow and frame ROI to be used next time.
        else:
            # Draw Shadow ROI on clone image.
            cv2.rectangle(img_roi_all, (left + 1, top + 1), (right - 1, bottom - 1), (0, 0, 255), 1)
            cv2.putText(img_roi_all, str(frame_index), (left + 3, bottom - 3), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, 1)

            # If the shadow image already exists, update it using a linear add.
            img_buff_str = './images/blocks/buffer/shadows/block' + str(frame_index) + '.jpg'
            if os.path.exists(img_buff_str):
                # Add the values of the current shadow image and previous shadow image.
                img_prev_buff = cv2.imread(img_buff_str)
                cv2.addWeighted(img_prev_buff, 1.0, img_buff, 1.0, 0.0, img_buff)

                # Store updated shadow image (don't update the old frame ROI).
                cv2.imwrite(img_buff_str, img_buff)
                print('Updated shadow image, \'block' + str(frame_index) + '.jpg\'')

            # Else if a black pixel was found write a new shadow image.
            elif dirty:
                cv2.imwrite(img_buff_str, img_buff)
                print('Wrote new shadow image, \'block' + str(frame_index) + '.jpg\'')

                # Store frame ROI image.
                img_buff_str = './images/blocks/buffer/frames/block' + str(frame_index) + '.jpg'
                img_roi = img_1[top:bottom, left:right]
                cv2.imwrite(img_buff_str, img_roi)

        # Increase frameblock index.
        frame_index += 1
        
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
