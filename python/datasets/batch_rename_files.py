# Batch renames an image directory to {:03d} format.
import argparse
import os as os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--inPath', default='./input/', help='input images data path')
parser.add_argument('--outPath', default='./output/', help='output images data path')

opt = parser.parse_args()
print(opt)

files = os.listdir(opt.inPath)
files.sort()
index = 1

for f in files:
    copyfile(opt.inPath + '{}'.format( f ), opt.outPath + '{:03d}.jpg'.format( index ))
    index += 1