import itertools
import scipy.io
import numpy as np
import argparse
import glob
import logging
import sys

import os

logger = logging.getLogger(__name__)


def load(features_file):
    image_features = []
    with open(features_file, 'r') as file:
        for line in file:
            items = line.split(' ')
            imgpath = items[0]
            features = list(map(float, items[1].split(',')))
            image_features.append((imgpath, features))
    return image_features


def merge_lists(image_features):
    return list(itertools.chain(*image_features))


def sort(image_features):
    result = sorted(image_features,
                    key=lambda imgpath_features: int(os.path.splitext(os.path.basename(imgpath_features[0]))[0]))
    return result


def convert_matrix(image_features):
    features = [features for imgpath, features in image_features]
    return np.array(features)


def extract(image_features, image_directory):
    filter_fun = lambda img_features: os.path.basename(os.path.split(img_features[0])[0]) == image_directory
    result = list(filter(filter_fun, image_features))
    if len(result) == 0:
        raise ValueError('no image features found for directory {}'.format(image_directory))
    return result


def save(features, features_directory):
    features_root_directory = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'features')
    target_filepath = os.path.join(features_root_directory, features_directory, 'alexnet-finetune-relu7.mat')
    scipy.io.savemat(target_filepath, {'features': features})
    return target_filepath


def process(image_features, save_directory):
    image_features = sort(image_features)
    features = convert_matrix(image_features)
    savepath = save(features, save_directory)
    logger.info('Saved {} to {}'.format(features.shape, savepath))


def main():
    type_dir_mapping = {'occluded': 'data_occlusion_klab325v2',
                        'whole': 'klab325_orig',
                        'lessOcclusion': 'lessOcclusion'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--file_pattern', type=str, default='images/predictions*.txt')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))

    features_files = glob.glob(args.file_pattern)
    logger.info('Loading from {} files'.format(len(features_files)))
    image_features = [load(features_file) for features_file in features_files]
    image_features = merge_lists(image_features)
    logger.info('Loaded features of size {}'.format(len(image_features)))

    def get_last_directory(path):
        return os.path.split(os.path.split(path)[0])[-1]

    for type in np.unique([get_last_directory(imagepath) for imagepath, _ in image_features]):
        assert type in type_dir_mapping
        features = extract(image_features, type)
        process(features, type_dir_mapping[type])


if __name__ == '__main__':
    main()
