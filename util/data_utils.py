import os
import shutil
import re


def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    else:
        print('File {} does not exist.'.format(path))


def preprocess_queries_data():
    IMAGE_DIR = 'queries'
    images = os.listdir(IMAGE_DIR)
    image_paths = [IMAGE_DIR + '/' + x for x in images]
    folders = list(filter(os.path.isdir, image_paths))
    paths_and_names = zip(folders, images)
    for x in paths_and_names:
        print(x)
        folder = x[0]
        name = x[1]
        sub_files = os.listdir(folder)
        for file in sub_files:
            match_obj = re.match(r'[\w-]+\.jpg|\w+\.img|\w+\.jpeg', file)
            if match_obj is None:
                remove(folder + '/' + file)


def preprocess_series_data():
    IMAGE_DIR = 'series'
    images = os.listdir(IMAGE_DIR)
    images = [IMAGE_DIR + '/' + x for x in images]
    folders = filter(os.path.isdir, images)
    for folder in folders:
        print(folder)
        # remove(folder + '/orb')
        # remove(folder + '/0')
        # remove(folder + '/1')
        # remove(folder + '/database.db')
        # remove(folder + '/reconstruction_log.txt')
        # remove(folder + '/objects')
        # remove(folder + '/old')
        os.system('mv ' + folder + '/old')

        # sub_images = os.listdir(folder + '/images')
        # for image in sub_images:
        #    os.rename(folder + '/images/' + image, folder + '/' + image)
        # os.rmdir(folder + '/images')
