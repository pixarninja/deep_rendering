import os as os
import glob as glob

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def clear_dir(path):
    if os.path.exists(path):
        filelist = glob.glob(path + '*')
        for file in filelist:
            os.remove(file)
    else:
        os.mkdir(path)