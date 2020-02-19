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

# Save the image with optional regions to disk.
def visualize_regions(path, image, regions):
    response = requests.get(image.url)
    if response.status_code == 200:
        os.path.dirname(path)
        with open(path, 'wb') as f:
            for chunk in response:
                f.write(chunk)
    
    img = Image.open(path)
    plt.imshow(img)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for region in regions:
        ax.add_patch(Rectangle((region.x, region.y), region.width, region.height, fill=False, edgecolor='red', linewidth=1))
        ax.text(region.x, region.y, region.phrase, style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    plt.savefig(path)
    plt.clf()

# Parse VisualGenome dataset from JSON.
def parse_json(image_path, attr_path, batch_size):
    # Open attribute JSON file.
    with open('./datasets/VisualGenome/attributes.json', 'r') as f:
        attributes = json.load(f)
        
    # Store ids for each image.
    ids = vg.get_image_ids_in_range(start_index=0, end_index=(batch_size - 1))
    w = 800.0;
    h = 600.0;
    
    for i, image_id in enumerate(ids):
        # Process image data.
        image = vg.get_image_data(id=image_id)
        response = requests.get(image.url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if not os.path.exists(image_path):
                os.makedirs(image_path)
                
            cv2.imwrite(image_path + '%03d.jpg' % (image_id), cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
                    
        # Process attribute data.
        attrs = attributes[i]['attributes']
        filtered_data = process_attributes(attrs, w, h)
        
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
        filtered_data.append(str(attr['object_id']) + ' ' + attr['names'][0] + ' ' + str(attr['h'] / h) + ' ' + str(attr['w'] / w) + ' ' + str(attr['x'] / w) + ' ' + str(attr['y'] / h))
    
    return filtered_data

base_path = './datasets/VisualGenome'
image_path = base_path + '/images/'
attr_path = base_path + '/attributes/'
batch_size = 512

# Parse images from source dataset.
parse_json(image_path, attr_path, batch_size)