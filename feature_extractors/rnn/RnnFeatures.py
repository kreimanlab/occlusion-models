import argparse
import functools
import os
import random

import h5py
import hdf5storage
import numpy as np
import scipy.io
from keras.layers import SimpleRNN
from keras.models import Sequential
from sklearn.cross_validation import KFold, train_test_split


class FeatureManager(object):
    def __init__(self, features_directory, input_filename, input_masked_filename, target_filename):
        super().__init__()
        self.features_directory = features_directory
        self.input_filename = input_filename
        self.input_masked_filename = input_masked_filename
        self.target_filename = target_filename

    def get_whole_features(self):
        return self.__load_features(self.target_filename)

    def get_occluded_features(self, data_selection=None):
        return self.__load_features(self.input_filename, data_selection)

    def get_feature_reshaper(self, mask_onset=None):
        if mask_onset is not None:
            mask_features = self.__load_features(self.input_masked_filename)
            return functools.partial(reshape_features_and_add_mask,
                                     mask_features=mask_features, mask_onset=mask_onset)
        else:
            return reshape_features

    def __load_features(self, filename, data_selection=None):
        filepath = os.path.join(self.features_directory, filename)
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
        hdf5storage.write({u'features': features}, path='.', filename=filepath,
                          matlab_compatible=True, truncate_existing=True)


def reshape_features(features, indices, timesteps=1):
    features = features[indices]
    features = np.resize(features, [features.shape[0], 1, features.shape[1]])
    return np.repeat(features, timesteps, 1)


def reshape_features_and_add_mask(features, indices, mask_features, timesteps=2, mask_onset=1):
    features = reshape_features(features, indices, timesteps=mask_onset)
    mask_features = reshape_features(mask_features, indices, timesteps=timesteps - mask_onset)
    return np.concatenate((features, mask_features), axis=1)


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


class Model:
    def __init__(self, X, Y, row_provider, feature_reshaper):
        assert Y.shape[1:] == X.shape[1:], \
            'feature sizes do not match: Y %s != X %s' % \
            (Y.shape[1:], X.shape[1:])
        self.original_shape = X
        self.X = self.flatten(X)
        self.Y = self.flatten(Y)
        self.row_provider = row_provider
        self.feature_reshaper = feature_reshaper
        self.model = self.__create_model()

    def __create_model(self):
        model = Sequential()
        rnn = SimpleRNN(name='RNN',
                        output_dim=self.X.shape[1], input_shape=(None, self.X.shape[1]),
                        activation='relu', init='identity', inner_init='zero',
                        return_sequences=True, consume_less="mem")
        model.add(rnn)
        rnn.trainable_weights.remove(rnn.W)
        rnn.trainable_weights.remove(rnn.b)
        rnn.non_trainable_weights += [rnn.W, rnn.b]  # input weights = identity matrix - we only care about recurrency
        model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
        return model

    def __train(self, model, train_indices, val_indices, num_epochs=10, batch_size=512, num_timesteps=5):
        initial_weights = model.get_weights()
        metrics = model.fit(reshape_features(self.X, train_indices, timesteps=num_timesteps),
                            reshape_features(self.Y, train_indices, timesteps=num_timesteps),
                            validation_data=(reshape_features(self.X, val_indices, timesteps=num_timesteps),
                                             reshape_features(self.Y, val_indices, timesteps=num_timesteps)),
                            batch_size=batch_size, nb_epoch=num_epochs, verbose=1)
        validation_losses = metrics.history['val_loss']
        best_epoch = np.array(validation_losses).argmin() + 1
        print('retraining on whole data up to best validation epoch %d' % best_epoch)
        model.reset_states()
        model.set_weights(initial_weights)
        model.fit(reshape_features(self.X, train_indices, timesteps=num_timesteps),
                  reshape_features(self.Y, train_indices, timesteps=num_timesteps),
                  batch_size=batch_size, nb_epoch=best_epoch, verbose=1)

    def __predict(self, indices, feature_reshaper, timesteps=6):
        Y = {}
        Y[0] = self.X[indices]
        predictions = self.model.predict(feature_reshaper(self.X, indices, timesteps=timesteps))
        for t in range(1, timesteps + 1):
            Y[t] = predictions[:, t - 1, :]
        return Y

    @staticmethod
    def __get_weights_file(kfold, suffix=''):
        weights_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'data', 'weights')
        if not os.path.isdir(weights_dir):
            os.mkdir(weights_dir)
        weights_filename = 'model_weights-kfold%d%s.hdf5' % (kfold, suffix)
        weights_file = os.path.join(weights_dir, weights_filename)
        return weights_file

    def cross_validate_prediction(self, train_epochs=10, max_timestep=6, weight_file_suffix=''):
        """
        for each kfold, train on subset of features and predict the rest.
        Ultimately predict all features by concatenating them for each kfold.
        """
        initial_model_weights = self.model.get_weights()
        predicted_features = np.zeros((max_timestep + 1,) + self.X.shape)
        num_kfold = 0
        for train_kfolds, val_kfolds, predict_kfolds in self.row_provider.get_kfolds():
            self.model.reset_states()
            self.model.set_weights(initial_model_weights)

            train_indices = self.row_provider.get_data_indices_from_kfolds(train_kfolds)
            validation_indices = self.row_provider.get_data_indices_from_kfolds(val_kfolds)
            predict_indices = self.row_provider.get_data_indices_from_kfolds(predict_kfolds)
            weights_file = self.__get_weights_file(num_kfold, weight_file_suffix)
            if os.path.isfile(weights_file):
                print('[kfold %d] using pre-trained weights %s' % (num_kfold, weights_file))
                self.model.load_weights(weights_file)
            else:
                print('[kfold %d] training...' % num_kfold)
                self.__train(self.model, train_indices, validation_indices, num_epochs=train_epochs)
                self.model.save_weights(weights_file, overwrite=True)
            print('[kfold %d] predicting...' % num_kfold)
            predicted_Y = self.__predict(predict_indices, self.feature_reshaper, timesteps=max_timestep)
            for timestep, prediction in predicted_Y.items():
                predicted_features[timestep, predict_indices, :] = prediction

            num_kfold += 1
        predicted_features = np.reshape(predicted_features, ((len(predicted_features),) + self.original_shape))
        return predicted_features

    @staticmethod
    def flatten(m):
        return np.reshape(m, (m.shape[0], np.prod(m.shape[1:])))


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
    parser.add_argument('--input_features', type=str, default='train/alexnet-Wfc6.mat',
                        help='features to predict')
    parser.add_argument('--input_data_selection', type=int, nargs=2,
                        help='what part of the input data to use')
    parser.add_argument('--input_features_masked', type=str, default=None,
                        help='input features for mask onset')
    parser.add_argument('--target_features', type=str, default='test/alexnet-fc7.mat',
                        help='target output features')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='how many epochs to search for optimal weights')
    parser.add_argument('--num_timesteps', type=int, default=6,
                        help='how many timesteps to run for prediction')
    parser.add_argument('--cross_validation', type=str, default='categories', choices=row_providers.keys(),
                        help='across what to validate')
    parser.add_argument('--add_whole_indices', action='store_true', default=False)
    parser.add_argument('--mask_onset', type=int, default=None,
                        help='at what timestep to use the masked input features')
    args = parser.parse_args()
    print('Running RNN with args', args)
    features_directory = args.features_directory
    num_epochs = args.num_epochs
    num_timesteps = args.num_timesteps
    row_provider = row_providers[args.cross_validation]
    if args.add_whole_indices:
        row_provider = RowsWithWhole(row_provider, pres)

    # load data
    print('loading features...')
    feature_manager = FeatureManager(features_directory,
                                     args.input_features, args.input_features_masked, args.target_features)
    whole_features = feature_manager.get_whole_features()
    occluded_features = feature_manager.get_occluded_features(args.input_data_selection)
    print('loaded features with shape whole=%s, occluded=%s' % (whole_features.shape, occluded_features.shape))
    assert whole_features.shape[1:] == occluded_features.shape[1:], \
        'feature sizes do not match: whole %s != occluded %s' % \
        (whole_features.shape[1:], occluded_features.shape[1:])
    aligned_whole_features = align_features(whole_features, occluded_features, pres)
    # TODO: conditionally offset with whole (np.concatenate(whole_features, occluded_features))
    if isinstance(row_provider, RowsAcrossImages):
        row_provider.set_num_images(occluded_features.shape[0])
    timestep_reshaper = feature_manager.get_feature_reshaper(args.mask_onset)
    # model
    print('creating model...')
    model = Model(occluded_features, aligned_whole_features, row_provider, timestep_reshaper)
    # run
    print('running...')
    predicted_Y = model.cross_validate_prediction(train_epochs=num_epochs, max_timestep=num_timesteps,
                                                  weight_file_suffix='-%s' % args.cross_validation)
    # save
    file_suffix = '-%s%s' % (args.cross_validation, '-mask%d' % args.mask_onset if args.mask_onset is not None else '')
    filename, file_extension = os.path.splitext(args.target_features)
    save_basename = '%s%s-rnn' % (os.path.basename(filename), file_suffix)
    save_dir_predicted = os.path.dirname(args.input_features)
    save_dir_whole = os.path.dirname(args.target_features)
    print('saving to %s...' % save_basename)
    for timestep in range(0, num_timesteps + 1):
        features = predicted_Y[timestep]
        assert features.shape == occluded_features.shape, \
            "predicted features shape %s does not match input features shape %s" \
            % (features.shape, occluded_features.shape)
        timestep_basename = '%s-t%d%s' % (save_basename, timestep, file_extension)
        feature_manager.save(features, os.path.join(save_dir_predicted, timestep_basename))
        feature_manager.save(whole_features, os.path.join(save_dir_whole, timestep_basename))


if __name__ == '__main__':
    random.seed(0)
    run_rnn()
