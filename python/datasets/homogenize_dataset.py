# Removes any attribute/image files which don't have a pair.
import argparse
import numpy as np
import os as os
import shutil as shutil
import pickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--trainingPath', default='C:/Users/wesha/Git/deep_rendering/python/datasets/Frame/training/', help='input training data path')
parser.add_argument('--outputPath', default='C:/Users/wesha/Git/deep_rendering/python/attngan/data/frame/text/', help='output training data path')
parser.add_argument('--picklePath', default='./', help='output filename pickle path')
parser.add_argument('--blockDim', type=int, default=64, help='dimension of frameblocks')
parser.add_argument('--probability', type=float, default=0.75, help='probability for training/testing image')

opt = parser.parse_args()
print(opt)

blocks_path = opt.trainingPath + str(opt.blockDim) + '/blocks/'
attributes_path = opt.trainingPath + str(opt.blockDim) + '/attributes/'
blocks = os.listdir(blocks_path)
blocks.sort()
training_list = []
testing_list = []
prob = opt.probability

for folder in blocks:
    # First check attributes folder path
    attrs_path = attributes_path + folder
    if os.path.exists(attrs_path) and len( os.listdir(attrs_path) ) > 0:
        # Next check if each attr file exists in blocks folder path
        block_files = os.listdir(blocks_path + folder)
        attr_files = os.listdir(attributes_path + folder)
        for f in attr_files:
            filename = f.replace('.txt', '.jpg')
            if filename not in block_files:
                # The block file wasn't found, prune its attribute pair
                file_path = attrs_path + '/' + f
                print('Removing attr file ' + file_path + '...')
                os.remove(file_path)
            else:
                # Otherwise add it to the filenames list and copy to output location
                flip = np.random.uniform(0, 1)
                if flip < prob:
                    training_list.append(folder + '/' + f.replace('.txt', ''))
                else:
                    testing_list.append(folder + '/' + f.replace('.txt', ''))
                shutil.copyfile(attrs_path + '/' + f, opt.outputPath + folder + '/' + f)
    else:
        print('Removing attr folder ' + attrs_path + '...')
        shutil.rmtree(attrs_path)

# Output the filenames list as a pickle
with open(opt.picklePath + 'train_filenames.pickle', 'wb') as pfile:
    pickle.dump(training_list, pfile, protocol=0)
with open(opt.picklePath + 'test_filenames.pickle', 'wb') as pfile:
    pickle.dump(testing_list, pfile, protocol=0)