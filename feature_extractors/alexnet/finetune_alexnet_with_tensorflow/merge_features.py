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


def save(features, target_filepath=os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                                                'data', 'features', 'data_occlusion_klab325v2',
                                                'alexnet-finetune-relu7.mat')):
    scipy.io.savemat(target_filepath, {'features': features})
    return target_filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--directory', type=str, default='images')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))

    features_files = glob.glob(os.path.join(args.directory, 'predictions*.txt'))
    logger.info('Loading from {} files'.format(len(features_files)))
    image_features = [load(features_file) for features_file in features_files]
    image_features = merge_lists(image_features)
    image_features = sort(image_features)
    features = convert_matrix(image_features)
    logger.info('Loaded features of size {}'.format(features.shape))
    savepath = save(features)
    logger.info('Saved to {}'.format(savepath))


if __name__ == '__main__':
    main()
