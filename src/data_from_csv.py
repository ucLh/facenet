import pandas as pd
import os
from facenet import ImageClass


def _process_frame(csv_dir, images_dir):
    df = pd.DataFrame.from_csv(csv_dir)
    df.drop('url', 1, inplace=True)
    df.sort_values(by='landmark_id', inplace=True)
    actual_images = os.listdir(images_dir)
    actual_images = list(map(lambda x: x.split('.')[0], actual_images))

    df = df.loc[df.index.isin(actual_images)]

    images = df.index.values
    images = list(map(lambda x: x + '.jpg', images))
    labels = df['landmark_id'].tolist()
    assert len(images) == len(labels)

    return images, labels, images_dir


def _get_facenet_dataset(images, labels, images_dir):
    dataset = list(zip(images, labels))

    old_label = labels[0]
    paths_list = []
    facenet_classes = []
    for x in dataset:
        image, label = x
        path = os.path.join(images_dir, image)

        if label == old_label:
            paths_list.append(path)
        else:
            facenet_classes.append(ImageClass(str(old_label), paths_list))
            old_label = label
            paths_list = [path]

    return facenet_classes


def get_facenet_dataset(csv_dir, images_dir):
    return _get_facenet_dataset(*_process_frame(csv_dir, images_dir))

# csv_dir = '/home/luch/Programming/Python/google-landmarks-dataset/v1/train.csv'
# images_dir = '/home/luch/Programming/Python/google-landmarks-dataset/v1/images'



