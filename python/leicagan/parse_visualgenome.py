#source: http://visualgenome.org/api/v0/api_beginners_tutorial.html

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visual_genome import api as vg
from PIL import Image
from io import BytesIO
import requests
import os
import json
import cv2
import numpy as np

# Data class for storing an image and it's associated JSON attributes.
class data():
    def __init__(self, image, attr): 
        self.image = image
        self.attr = attr
          
    def toString(self):
        print("Image: ", self.image )
        print("Attributes: ", self.attr )

# Save the image with attributes to disk.
def visualize_regions(path, i):
    # Initialize path variables.
    images_path = path + '/images/'
    attr_path = path + '/attributes/'

    # Setup main loop to process all images in an animation.
    images = os.listdir(images_path)
    images.sort()

    # Obtain attributes from written file.
    attrs = []
    with open(attr_path + '%03d.dat' % (i + 1), 'r') as f:
        for line in f:
            values = line.split()
            attrs.append(values)

    # Select an image index and set output image location.
    image = images[i]
    img_str = path + '/' + str(i + 1) + '_blocks.jpg'
    image_str_out = path + '/' + str(i + 1) + '_attrs.jpg'
    
    # Load image and attributes.
    img = cv2.cvtColor(cv2.imread(img_str), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    plt.imshow(img)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for attr in attrs:
        ax.add_patch(Rectangle((float(attr[4]) * w, float(attr[5]) * h), float(attr[3]) * w, float(attr[2]) * h, fill=False, edgecolor='red', linewidth=1))
        ax.text(float(attr[4]) * w, float(attr[5]) * h, attr[1], style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(image_str_out)
    plt.clf()

# Save the image with attributes and frameblocks to disk.
def visualize_frameblocks(path, i):        
    # Overlay frameblocks before plotting attributes.
    block_dim = 64
    block_offset = 1

    # Initialize path variables.
    images_path = path + '/images/'
    attr_path = path + '/attributes/'
    
    training_prefix = path + '/training/'
    block_prefix = training_prefix + str(block_dim)
    training_path = block_prefix + '/blocks/'

    # Setup main loop to process all images in an animation.
    images = os.listdir(images_path)
    images.sort()

    # Initialize seed variables.
    image = images[i]
    image_str_out = path + '/' + str(i + 1) + '_blocks.jpg'
    block_index = 1
    img_str = images_path + image

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

            # Draw Block ROI on clone image.
            cv2.rectangle(img, (left + 1, top + 1), (right - 1, bottom - 1), (255, 0, 0), 1)
            
            # Increase imageblock index.
            block_index += 1

            # Shift horizontally.
            left += int(block_dim / block_offset)
            right += int(block_dim / block_offset)

        # Shift vertically.
        top += int(block_dim / block_offset)
        bottom += int(block_dim / block_offset)
        left = 0
        right = block_dim

    cv2.imwrite(image_str_out, img)

# Parse VisualGenome dataset from JSON.
def parse_json(image_path, attr_path, batch_size):
    # Open attribute JSON file.
    print('Loading JSON file...')
    with open('./datasets/VisualGenome/attributes.json', 'r') as f:
        attributes = json.load(f)
        
    # Store ids for each image.
    ids = vg.get_image_ids_in_range(start_index=0, end_index=(batch_size - 1))
    w = 800.0
    h = 600.0
    
    for i, image_id in enumerate(ids):
        print('Processing ID ' + str(image_id) + '...')
        
        # Process image data.
        image = vg.get_image_data(id=image_id)
        response = requests.get(image.url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            h, w = np.array(img).shape[:2]
            if not os.path.exists(image_path):
                os.makedirs(image_path)
                
            cv2.imwrite(image_path + '%03d.jpg' % (image_id), cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                    
        # Process attribute data.
        attrs = attributes[i]['attributes']
        filtered_data = process_attributes(attrs, float(w), float(h))
        
        #regions = vg.get_region_descriptions_of_image(id=image_id)
        #filtered_data = process_regions(regions, w, h)

        if not os.path.exists(attr_path):
            os.makedirs(attr_path)
        with open(attr_path + '%03d.dat' % (image_id), 'w') as f:
            for n, line in enumerate(filtered_data):
                if n > 0:
                    f.write('\n' + line)
                else:
                    f.write(line)
                    
# Process region information.
def process_regions(regions, w, h):
    filtered_data = []
    for region in regions:
        filtered_data.append(str(region.id) + ' ' + region.phrase.replace(" ", "") + ' ' + str(region.height / h) + ' ' + str(region.width / w) + ' ' + str(region.x / w) + ' ' + str(region.y / h))
    
    print(filtered_data)
    return filtered_data

# Process attribute information.
def process_attributes(attrs, w, h):
    filtered_data = []
    for attr in attrs:
        filtered_data.append(str(attr['object_id']) + ' ' + attr['names'][0].replace(" ", "") + ' ' + str(attr['h'] / h) + ' ' + str(attr['w'] / w) + ' ' + str(attr['x'] / w) + ' ' + str(attr['y'] / h))
    
    return filtered_data

base_path = './datasets/VisualGenome'
image_path = base_path + '/images/'
attr_path = base_path + '/attributes/'
batch_size = 512

# Parse images from source dataset.
#parse_json(image_path, attr_path, batch_size)

# Save a sample image.
base_path = './datasets/VisualGenome'
visualize_frameblocks(base_path, 96)
visualize_regions(base_path, 96)