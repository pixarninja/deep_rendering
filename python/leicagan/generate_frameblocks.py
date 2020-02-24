import argparse
import cv2
import os
import numpy as np
import utils

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y
        
    def toString(self):
        return str(self.x) + ", " + str(self.y)

def generate_imageblocks(path):
    # Initialize seed variables.
    block_dim = opt.blockDim
    block_offset = opt.blockOffset

    # Initialize path variables.
    images_path = path + '/images/'
    attr_path = path + '/attributes/'
    
    training_prefix = path + '/training/'
    block_prefix = training_prefix + str(block_dim)
    training_path = block_prefix + '/blocks/'
    attribute_path = block_prefix + '/attributes/'

    # Delete previously output imageblocks, and buffer shadows and buffer images.
    utils.make_dir(training_prefix)
    utils.make_dir(block_prefix)
    utils.clear_dir(training_path)
    utils.clear_dir(attribute_path)

    # Setup main loop to process all images in an animation.
    images = os.listdir(images_path)
    images.sort()

    # Process each image.
    for image_index in range(0, len(images)):
        if image_index > 0:
            exit()
        # Obtain attributes from written file.
        attrs = []
        with open(attr_path + '%03d.dat' % (image_index + 1), 'r') as f:
            for line in f:
                values = line.split()
                attrs.append(values)

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
                    
                # Check bounds for each attribute for each block.
                # Indexed as bound[ [left, right], [top, bottom] ].
                attrs_inside_roi = []
                for attr in attrs:
                    h = float(attr[2])
                    w = float(attr[3])
                    x = float(attr[4])
                    y = float(attr[5])
                    
                    l1 = Point(left / float(width), bottom / float(height)) 
                    r1 = Point(right / float(width), top / float(height)) 
                    l2 = Point(x, y + h) 
                    r2 = Point(x + w, y)
                    
                    if overlap( l1, r1, l2, r2 ):
                        # Calculate offsets based on coordinates of frameblock
                        attr[2] = str(h)
                        attr[3] = str(w)
                        attr[4] = str((x - left))
                        attr[5] = str((y - top))
                        attrs_inside_roi.append(attr)

                if attrs_inside_roi != []:
                    # ROI pixel processing
                    print(str(block_index) + ". imageblock: (" + str(left) + ", " + str(top) + "), (" + str(right) + ", " + str(bottom) + "))")

                    # Store window contents as image.
                    img_str_out = training_path + image_str_out + block_str_out
                    img_roi = img[top:bottom, left:right]
                    cv2.imwrite(img_str_out + '.jpg', img_roi)
                    
                    # Output found attributes to file.
                    attr_str_out = attribute_path + image_str_out + block_str_out
                    with open(attr_str_out + '.dat', 'w') as f:
                        for n, line in enumerate(attrs_inside_roi):
                            if n > 0:
                                f.write('\n' + ' '.join(line))
                            else:
                                f.write(' '.join(line))

                # Increase imageblock index.
                block_index += 1
                block_str_out = 'block' + str(block_index + 1)
                
                # Shift horizontally.
                left += int(block_dim / block_offset)
                right += int(block_dim / block_offset)

            # Shift vertically.
            top += int(block_dim / block_offset)
            bottom += int(block_dim / block_offset)
            print(str(bottom) + ', ' + str(height))
            left = 0
            right = block_dim

# Helper function to determine overlapping rectangles.
def overlap(l1, r1, l2, r2): 
    # If one rectangle is on left side of other 
    if(l1.x > r2.x or l2.x > r1.x): 
        return False
  
    # If one rectangle is above other 
    if(l1.y < r2.y or l2.y < r1.y): 
        return False
  
    return True
            
parser = argparse.ArgumentParser()
parser.add_argument('--blockDim', type=int, default=64, help='dimension of imageblocks')
parser.add_argument('--blockOffset', type=int, default=1, help='offset for blocks, > 1 blocks will overlap')

opt = parser.parse_args()
print(opt)

l1 = Point(0.48, 0)
r1 = Point(0.56, 0.106666666667)
l2 = Point(0.4775, 0.615)
r2 = Point(0.5449999, 0.82833333)
            
# Generate frameblocks off of parsed images.
base_path = './datasets/VisualGenome'
generate_imageblocks(base_path)