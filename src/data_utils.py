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
    IMAGE_DIR = 'series'
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
            match_obj = re.match(r'[\w-]+\.jpg|[\w-+]\.img|[\w-+]\.jpeg|[\w-]+\.JPG', file)
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


def move_needed_classes():
    needed_classes_dir = 'queries'
    classes_dir = 'weather-localization-training-set'
    needed_classes = os.listdir(needed_classes_dir)
    classes = os.listdir(classes_dir)
    needed_classes = list(map(lambda c: c.split('__')[0], needed_classes))

    for class_ in classes:
        tmp = class_.split('__')[0]
        if tmp in needed_classes:
            print('mv ' + classes_dir + '/' + class_ +' series_for_test/')
            os.system('mv ' + classes_dir + '/' + class_ +' series_for_test/')


def get_folder2label_dict():

    def get_class(dir_name):
        return dir_name.split('__')[0]

    IMAGE_DIR = '../datasets/queries'
    image_names = os.listdir(IMAGE_DIR)
    image_names = set(map(get_class, image_names))
    image_names = list(image_names)
    image_names.sort()

    labels_dict = {get_class(image_names[i]): i for i in range(len(image_names))}

    return labels_dict
