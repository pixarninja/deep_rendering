import os as os
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

block_dim = 32
block_offset = 1

# Initialize seed variables.
img_str_1 = './images/samples/frame_1_complex.jpg'
img_str_2 = './images/samples/frame_3_complex.jpg'
img_str_out = './images/output_complex.jpg'
img_str_roi = './images/frameblock_roi_complex.jpg'
os.system('rm -rf %s' % './images/blocks/buffer/frames/')
os.mkdir('./images/blocks/buffer/frames/')

# Delete previously calculated frames, buffers, and shadows.
os.system('rm -rf %s' % './images/blocks/complex/')
os.mkdir('./images/blocks/complex/')
os.system('rm -rf %s' % './images/blocks/buffer/frames/')
os.mkdir('./images/blocks/buffer/frames/')
os.system('rm -rf %s' % './images/blocks/buffer/shadows/')
os.mkdir('./images/blocks/buffer/shadows/')

img_1 = cv2.cvtColor(cv2.imread(img_str_1), cv2.COLOR_BGR2RGB)
height_1, width_1 = img_1.shape[:2]
img_2 = cv2.cvtColor(cv2.imread(img_str_2), cv2.COLOR_BGR2RGB)
height_2, width_2 = img_2.shape[:2]

# Choose smallest boundaries
height = height_1
width = width_1
if height_1 > height_2:
    height = height_2
if width_1 > width_2:
    width = width_2

# Plot images for reference.
plt.subplot(121),plt.imshow(img_1),plt.title('Input Image 1')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_2),plt.title('Input Image 2')
plt.xticks([]), plt.yticks([])

img_out = np.ones((height, width, 3), np.uint8)
total_pixel_sum = 0

# Loop through each pixel.
for y in range(0, height):
    for x in range(0, width):
        #print(img_sample_1[y, x], img_sample_2[y, x])
        pixel_out = abs(img_1[y, x] ^ img_2[y, x])
        #print(pixel_out)
        pixel_sum = int(pixel_out[0]) + int(pixel_out[1]) + int(pixel_out[2])
        if pixel_sum > 255:
            bw_val = 255
        else:
            bw_val = np.uint8(pixel_sum)
        total_pixel_sum += bw_val
        bw_val = 255 - bw_val
        img_out[y, x] = bw_val

# Write image and plot to screen.
cv2.imwrite(img_str_out, img_out)
plt.subplot(111),plt.imshow(img_out),plt.title('Calculated Image for Frameblock Generation')
plt.xticks([]), plt.yticks([])

# Calculate the pixel_ratio.
print("Total pixel sum: " + str(total_pixel_sum))
pixel_ratio = total_pixel_sum * 1.0 / (255 * width * height)
print("Pixel ratio: " + str(pixel_ratio))

# Create a clone of input image and draw ROIs on top of it.
img_roi_all = cv2.imread(img_str_out)

# Create sliding window.
left = 0
right = block_dim
top = 0
bottom = block_dim
frame_index = 1
pixel_sum = 0
cap = np.power(block_dim, 2) * 255 * pixel_ratio
print("Cap found: " + str(cap))

# Delete previously found buffers but leave images
os.system('rm -rf %s' % './images/blocks/buffer/shadows/')
os.mkdir('./images/blocks/buffer/shadows/')
os.system('rm -rf %s' % './images/blocks/complex/')
os.mkdir('./images/blocks/complex/')

# Find the Region Of Interest (ROI).
while bottom < height:
    while right < width:
        found_x = False
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
                if img_buff[y - top - 1, x - left - 1][0] > 255:
                    img_buff[y - top - 1, x - left - 1] = 255
                pixel_sum += img_buff[y - top - 1, x - left - 1][0]
                
                # Test if the cap was met.
                if pixel_sum >= cap:
                    # Draw ROI on clone image.
                    cv2.rectangle(img_roi_all, (left, top), (right, bottom), (255, 0, 0), 1)
                    print(str(frame_index) + ". Sum: " + str(pixel_sum) + ", Frameblock: (" + str(left) + ", " + str(top) + "), (" + str(right) + ", " + str(bottom) + "))")
                    
                    # Store window contents as image.
                    img_roi_1 = img_1[top:bottom, left:right]
                    img_roi_2 = img_2[top:bottom, left:right]
                    cv2.imwrite('./images/blocks/complex/block' + str(frame_index) + '_1.jpg', img_roi_1)
                    cv2.imwrite('./images/blocks/complex/block' + str(frame_index) + '_2.jpg', img_roi_2)
                    
                    # Exit both for loops.
                    found_x = True
                    break
            if found_x:
                break
        # If frameblock was used delete buffer file, else export buffer and frame ROI.
        if found_x:
            if os.path.exists(img_buff_str):
                os.remove(img_buff_str)
        else:
            cv2.imwrite(img_buff_str, img_buff)
            img_buff_str = './images/blocks/buffer/frames/block' + str(frame_index) + '_1.jpg'
            if not os.path.exists(img_buff_str):
                img_roi_buff = img_1[top:bottom, left:right]
                cv2.imwrite(img_buff_str, img_roi_buff)
            img_buff_str = './images/blocks/buffer/frames/block' + str(frame_index) + '_2.jpg'
            if not os.path.exists(img_buff_str):
                img_roi_buff = img_2[top:bottom, left:right]
                cv2.imwrite(img_buff_str, img_roi_buff)
        frame_index += 1
        
        # Shift horizontally
        left += int(block_dim / block_offset)
        right += int(block_dim / block_offset)
    # Shift vertically
    top += int(block_dim / block_offset)
    bottom += int(block_dim / block_offset)
    left = 0
    right = block_dim

cv2.imwrite(img_str_roi, img_roi_all)
plt.subplot(111),plt.imshow(img_roi_all),plt.title('Frameblocks Selected (' + str(frame_index - 1) + ')')
plt.xticks([]), plt.yticks([])

frame_index = 1
img_block_str = './images/blocks/complex/block' + str(frame_index) + '_1.jpg'
while not os.path.exists(img_block_str):
    frame_index += 1
    img_block_str = './images/blocks/complex/block' + str(frame_index) + '_1.jpg'
img_block_1 = cv2.cvtColor(cv2.imread(img_block_str), cv2.COLOR_BGR2RGB)
img_block_str = './images/blocks/complex/block' + str(frame_index) + '_2.jpg'
img_block_2 = cv2.cvtColor(cv2.imread(img_block_str), cv2.COLOR_BGR2RGB)
plt.subplot(121),plt.imshow(img_block_1),plt.title('Frameblock 1 From Image 1')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_block_2),plt.title('Frameblock 1 From Image 2')
plt.xticks([]), plt.yticks([])
plt.show()

frame_index += 1
img_block_str = './images/blocks/complex/block' + str(frame_index) + '_1.jpg'
while not os.path.exists(img_block_str):
    frame_index += 1
    img_block_str = './images/blocks/complex/block' + str(frame_index) + '_1.jpg'
img_block_1 = cv2.cvtColor(cv2.imread(img_block_str), cv2.COLOR_BGR2RGB)
img_block_str = './images/blocks/complex/block' + str(frame_index) + '_2.jpg'
img_block_2 = cv2.cvtColor(cv2.imread(img_block_str), cv2.COLOR_BGR2RGB)
plt.subplot(121),plt.imshow(img_block_1),plt.title('Frameblock 2 From Image 1')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_block_2),plt.title('Frameblock 2 From Image 2')
plt.xticks([]), plt.yticks([])
plt.show()
