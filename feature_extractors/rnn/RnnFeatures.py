import argparse
import os
import random

import functools

import h5py
import hdf5storage
import numpy as np
import scipy.io
from keras.layers import SimpleRNN
from keras.models import Sequential
from sklearn.cross_validation import KFold, train_test_split


class FeatureManager(object):
    def __init__(self, features_directory, input_filename, target_filename):
        super().__init__()
        self.features_directory = features_directory
        self.input_filename = input_filename
        self.target_filename = target_filename

    def get_whole_features(self):
        filepath = os.path.join(self.features_directory, self.target_filename)
        return self.__load_features(filepath)

    def get_occluded_features(self, data_selection=None):
        filepath = os.path.join(self.features_directory, self.input_filename)
        return self.__load_features(filepath, data_selection)

    def load_mask_features(self):
        raise NotImplementedError()

    def get_feature_reshaper(self, mask_onset):
        if mask_onset >= 0:
            mask_features = self.load_mask_features()
            return functools.partial(reshape_features_and_add_mask,
                                     mask_features=mask_features, mask_onset=mask_onset)
        else:
            return reshape_features

    @staticmethod
    def __load_features(filepath, data_selection=None):
        assert os.path.isfile(filepath), "file does not exist: %s" % os.path.realpath(filepath)
        if filepath.endswith('.txt'):
            features = np.loadtxt(filepath)
            if data_selection is not None:
                features = features[range(data_selection[0], data_selection[1]), :]
            return features
        if filepath.endswith('.mat'):
            try:
                with h5py.File(filepath, 'r') as file:
                    features = file['features']
                    features = np.array(features).transpose()  # Matlab dimension-ordering is column-major (reverse)
                    if data_selection is not None:
                        features = features[range(data_selection[0], data_selection[1]), :]
                    return features
            except (ValueError, OSError):
                data = scipy.io.loadmat(filepath)
                features = np.array(data['features'])
                if data_selection is not None:
                    features = features[range(data_selection[0], data_selection[1]), :]
                return features
        raise ValueError('Unknown file extension in features filename %s' % filepath)

    def save(self, features, filename):
        filepath = os.path.join(self.features_directory, filename)
        hdf5storage.write({u'features': features}, path=filepath, matlab_compatible=True)


class RowProvider:
    def get_kfolds(self):
        """
        :return: triple train_kfolds, validation_kfolds, test_kfolds
        """
        pass

    def get_data_indices_from_kfolds(self, kfold_values):
        pass


class RowsAcrossImages(RowProvider):
    def __init__(self):
        self.num_images = None

    def set_num_images(self, num_images):
        self.num_images = num_images

    def get_kfolds(self, num_kfolds=5, validation_split=0.1):
        assert self.num_images is not None, "num_images is not set. call set_num_images(<num_images>) first"
        for train_val_rows, test_rows in KFold(self.num_images, num_kfolds):
            train_rows, val_rows = train_test_split(train_val_rows, test_size=validation_split)
            yield train_rows, val_rows, test_rows

    def get_data_indices_from_kfolds(self, rows):
        return rows


class RowsAcrossObjects(RowProvider):
    def __init__(self, pres):
        self.pres = pres

    def get_kfolds(self, num_kfolds=5, validation_split=0.1):
        unique_pres = list(set(self.pres))
        kfold = KFold(len(unique_pres), num_kfolds)
        for train_val_object_indices, predict_object_indices in kfold:
            train_object_indices, val_object_indices = train_test_split(train_val_object_indices,
                                                                        test_size=validation_split)

            train_objects = self.get_objects(train_object_indices)
            val_objects = self.get_objects(val_object_indices)
            predict_objects = self.get_objects(predict_object_indices)
            yield (train_objects, val_objects, predict_objects)

    def get_data_indices_from_kfolds(self, objects):
        return [i for i in range(len(self.pres)) if self.pres[i] in objects]

    def get_objects(self, pres_indices):
        return [self.pres[i] for i in pres_indices]


class RowsAcrossCategories(RowProvider):
    def __init__(self):
        self.categories = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  "../../../data/data_occlusion_klab325v2-categories.txt"), dtype='int')

    def get_kfolds(self, num_kfolds=5, validation_split=0.1):
        unique_categories = list(set(self.categories))
        assert num_kfolds == len(unique_categories), \
            "num_kfolds = %d and len(unique_categories) = %d do not match" % (num_kfolds, len(unique_categories))
        kfold = KFold(len(unique_categories), num_kfolds)
        for train_val_category_indices, predict_category_indices in kfold:
            train_category_indices, val_category_indices = train_test_split(train_val_category_indices,
                                                                            test_size=validation_split)
            train_categories = [unique_categories[i] for i in train_category_indices]
            val_categories = [unique_categories[i] for i in val_category_indices]
            predict_categories = [unique_categories[i] for i in predict_category_indices]
            yield (train_categories, val_categories, predict_categories)

    def get_data_indices_from_kfolds(self, categories):
        return [i for i in range(len(self.categories)) if self.categories[i] in categories]


class RowsWithWhole(RowProvider):
    def __init__(self, inner_provider, pres):
        self.inner_provider = inner_provider
        self.pres = pres

    def get_kfolds(self):
        return self.inner_provider.get_kfolds

    def get_data_indices_from_kfolds(self, kfold_values):
        indices = self.inner_provider.get_data_indices_from_kfolds(kfold_values)
        pres = list(set(self.pres[indices]))
        return np.concatenate(([i + 325 for i in indices], pres))


def create_model(feature_size):
    model = Sequential()
    model.add(SimpleRNN(output_dim=feature_size, input_shape=(None, feature_size),
                        activation='relu',
                        return_sequences=True, name='RNN'))
    model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
    return model


def train(model, X_train, Y_train, X_val, Y_val, num_epochs=10, batch_size=512):
    assert len(X_train) == len(Y_train), \
        "len(X_train) = %d and len(Y_train) = %d do not match" % (len(X_train), len(Y_train))
    assert len(X_val) == len(Y_val), \
        "len(X_val) = %d and len(Y_val) = %d do not match" % (len(X_val), len(Y_val))
    initial_weights = model.get_weights()
    metrics = model.fit(reshape_features(X_train), reshape_features(Y_train),
                        validation_data=(reshape_features(X_val), reshape_features(Y_val)),
                        batch_size=batch_size, nb_epoch=num_epochs, verbose=1)
    validation_losses = metrics.history['val_loss']
    best_epoch = np.array(validation_losses).argmin() + 1
    print('retraining on whole data up to best validation epoch %d' % best_epoch)
    model.reset_states()
    model.set_weights(initial_weights)
    model.fit(reshape_features(X_train), reshape_features(Y_train),
              batch_size=batch_size, nb_epoch=best_epoch, verbose=1)


def flatten(m):
    return np.reshape(m, (m.shape[0], np.prod(m.shape[1:])))


def reshape_features(features, timesteps=1):
    features = np.resize(features, [features.shape[0], 1, features.shape[1]])
    return np.repeat(features, timesteps, 1)


def reshape_features_and_add_mask(features, mask_features, timesteps=2, mask_timestep_onset=1):
    features = reshape_features(features, mask_timestep_onset)
    mask_features = reshape_features(mask_features, timesteps - mask_timestep_onset)
    return np.concatenate((features, mask_features), axis=1)


def predict(model, X, feature_reshaper, timesteps=6):
    Y = {}
    Y[0] = X
    predictions = model.predict(feature_reshaper(X, timesteps))
    for t in range(1, timesteps + 1):
        Y[t] = predictions[:, t - 1, :]
    return Y


def align_features(whole_features, occluded_features, pres):
    assert len(pres) == len(occluded_features), \
        "len(pres) = %d and len(occluded_features) = %d do not match" % (len(pres), len(occluded_features))
    assert whole_features.shape[1:] == occluded_features.shape[1:], \
        "whole_features.shape[1:] = %s and occluded_features.shape[1:] = %s do not match" % \
        ((whole_features.shape[1:],), (occluded_features.shape[1:],))
    aligned_whole = np.zeros(occluded_features.shape)
    for i in range(len(occluded_features)):
        corresponding_whole = int(pres[i])
        aligned_whole[i, :] = whole_features[corresponding_whole - 1, :]
    return aligned_whole


def get_weights_file(kfold, suffix=''):
    weights_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'data', 'weights')
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    weights_filename = 'model_weights-kfold%d%s.hdf5' % (kfold, suffix)
    weights_file = os.path.join(weights_dir, weights_filename)
    return weights_file


def cross_validate_prediction(model, X, Y, row_provider, feature_reshaper,
                              train_epochs=10, max_timestep=6, file_suffix=''):
    """
    for each kfold, train on subset of features and predict the rest.
    Ultimately predict all features by concatenating them for each kfold.
    """
    initial_model_weights = model.get_weights()
    predicted_features = np.zeros((max_timestep + 1,) + X.shape)
    num_kfold = 0
    for train_kfolds, val_kfolds, predict_kfolds in row_provider.get_kfolds():
        model.reset_states()
        model.set_weights(initial_model_weights)

        train_indices = row_provider.get_data_indices_from_kfolds(train_kfolds)
        validation_indices = row_provider.get_data_indices_from_kfolds(val_kfolds)
        predict_indices = row_provider.get_data_indices_from_kfolds(predict_kfolds)
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_val, Y_val = X[validation_indices], Y[validation_indices]
        X_predict = X[predict_indices]
        weights_file = get_weights_file(num_kfold, file_suffix)
        if os.path.isfile(weights_file):
            print('[kfold %d] using pre-trained weights %s' % (num_kfold, weights_file))
            model.load_weights(weights_file)
        else:
            print('[kfold %d] training...' % num_kfold)
            train(model, X_train, Y_train, X_val, Y_val, num_epochs=train_epochs)
            model.save_weights(weights_file, overwrite=True)
        print('[kfold %d] predicting...' % num_kfold)
        predicted_Y = predict(model, X_predict, feature_reshaper, timesteps=max_timestep)
        for timestep, prediction in predicted_Y.items():
            predicted_features[timestep, predict_indices, :] = prediction

        num_kfold += 1
    return predicted_features


def run_rnn():
    # params - fixed
    pres = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "../../../data/data_occlusion_klab325v2-pres.txt"), dtype='int')
    row_providers = {'images': RowsAcrossImages(),
                     'objects': RowsAcrossObjects(pres),
                     'categories': RowsAcrossCategories()}
    # params - command line
    parser = argparse.ArgumentParser(description='Train and predict whole features from occluded ones')
    parser.add_argument('--features_directory', type=str,
                        help='directory containing features to load / target directory for predicted features')
    parser.add_argument('--input_features', type=str, default='train/alexnet-WFc6.mat',
                        help='features to predict')
    parser.add_argument('--input_data_selection', type=int, nargs=2,
                        help='what part of the input data to use')
    parser.add_argument('--target_features', type=str, default='test/alexnet-fc7.mat',
                        help='target output features')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='how many epochs to search for optimal weights')
    parser.add_argument('--num_timesteps', type=int, default=6,
                        help='how many timesteps to run for prediction')
    parser.add_argument('--cross_validation', type=str, default='categories',
                        choices=row_providers.keys(), help='across what to validate')
    parser.add_argument('--add_whole_indices', action='store_true', default=False)
    parser.add_argument('--mask_onset', type=int, default=-1, choices=range(1, 6),
                        help='across what to validate')
    args = parser.parse_args()
    print('Running RNN with args', args)
    features_directory = args.features_directory
    num_epochs = args.num_epochs
    num_timesteps = args.num_timesteps
    row_provider = row_providers[args.cross_validation]
    file_suffix = '-%s%s' % (args.cross_validation, '-mask%d' % args.mask_onset if args.mask_onset >= 0 else '')
    if args.add_whole_indices:
        row_provider = RowsWithWhole(row_provider, pres)

    # load data
    print('loading features...')
    feature_manager = FeatureManager(features_directory, args.input_features, args.target_features)
    Y = feature_manager.get_whole_features()
    X = feature_manager.get_occluded_features(args.input_data_selection)
    assert Y.shape[1:] == X.shape[1:], \
        'feature sizes do not match: whole %s != occluded %s' % \
        (Y.shape[1:], X.shape[1:])
    Y = align_features(Y, X, pres)
    original_X_shape = X.shape[1:]
    print('X shape: %s' % (original_X_shape,))
    # flatten
    X, Y = flatten(X), flatten(Y)
    # TODO: conditionally offset with whole (np.concatenate(Y, X))
    if isinstance(row_provider, RowsAcrossImages):
        row_provider.set_num_images(X.shape[0])
    timestep_reshaper = feature_manager.get_feature_reshaper(args.mask_onset)
    # model
    feature_size = X.shape[1]
    print('creating model with collapsed feature size %d...' % feature_size)
    model = create_model(feature_size)
    # run
    print('running...')
    predicted_Y = cross_validate_prediction(model, X, Y, row_provider, timestep_reshaper,
                                            train_epochs=num_epochs, max_timestep=num_timesteps,
                                            file_suffix=file_suffix)
    # save
    print('saving...')
    for timestep in range(0, num_timesteps + 1):
        features = predicted_Y[timestep]
        features = np.reshape(features, X.shape)
        filename = '%s%s-rnn-t%d' % (args.target_features, file_suffix, timestep)
        feature_manager.save(features, filename)


if __name__ == '__main__':
    random.seed(0)
    run_rnn()
