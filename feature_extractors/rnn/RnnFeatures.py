import argparse
import functools
import os
import random
import warnings

import h5py
import hdf5storage
import numpy as np
import scipy.io
from keras import backend as K
from keras.applications import ResNet50, InceptionV3, VGG16, VGG19
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cross_validation import KFold, train_test_split

try:
    from .SimplerRnn import SimpleURNN
except (SystemError, ModuleNotFoundError):
    from SimplerRnn import SimpleURNN

TESTRUN = False


class FeatureManager(object):
    def __init__(self, features_directory, input_filename, input_masked_filename, target_filename):
        super().__init__()
        self.features_directory = features_directory
        self.input_filename = input_filename
        self.input_masked_filename = input_masked_filename
        self.target_filename = target_filename

    def get_whole_features(self):
        return self.load_features(self.target_filename)

    def get_occluded_features(self, data_selection=None):
        return self.load_features(self.input_filename, data_selection)

    def get_masked_features(self):
        return self.load_features(self.input_masked_filename)

    def load_features(self, filename, data_selection=None):
        if filename is None:
            warnings.warn('features filename is None - returning None')
            return None

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

                    if isinstance(features[0][0], h5py.h5r.Reference):
                        features = features[0]
                        _features = []
                        for i in range(features.shape[0]):
                            # get a reference to the actual value
                            reference = features[i]
                            # retrieve actual data
                            data = file[reference].value
                            _features.append(data.transpose())
                        features = np.array(_features)
                    else:
                        features = np.array(features).transpose()  # Matlab dimension-ordering is column-major (reverse)

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


class PlainModel(object):
    def get_feature_reshaper(self, mask_onset=None):
        if mask_onset is not None:
            raise NotImplementedError('feature masking is not implemented in this feed-forward model')
        else:
            def reshape_features(X, indices, timesteps):
                if timesteps != 1:
                    raise ValueError('no reshaping, so timesteps needs to be equal to 1')
                return X[indices]

            return reshape_features


class KerasModel(PlainModel):
    def __init__(self, model_constructor, preprocess_input, feature_manager=None):
        super(KerasModel, self).__init__()
        self.model_constructor = model_constructor
        self.preprocess_input = preprocess_input

    def get_feature_reshaper(self, mask_onset=None):
        reshape = super(KerasModel, self).get_feature_reshaper(mask_onset)

        def reshape_and_preprocess(X, indices, timesteps):
            reshaped_X = reshape(X, indices, timesteps)
            preprocessed_X = self.preprocess_input(reshaped_X.astype('float64'))
            return preprocessed_X

        return reshape_and_preprocess

    def __call__(self, shape, include_top=False):
        return self.model_constructor(include_top=include_top, weights="imagenet", input_shape=shape)


class KerasTrainModel(KerasModel):
    def __call__(self, shape, include_top=True):
        return super(KerasTrainModel, self).__call__(shape, include_top=include_top)


class FeedForwardModel(PlainModel):
    def __init__(self, _):
        pass

    def __call__(self, num_features):
        model = Sequential()
        model.add(Dense(output_dim=num_features, input_dim=num_features,
                        activation='relu', init='zero'))
        model.add(Dense(output_dim=num_features,
                        activation='relu', init='zero'))
        model.add(Dense(output_dim=num_features,
                        activation='relu', init='zero'))
        model.add(Dense(output_dim=num_features,
                        activation='relu', init='zero'))
        model.add(Dense(output_dim=num_features,
                        activation='relu', init='zero'))

        def unfold_layer_outputs(X, *args, **kwargs):
            Ys = {}
            print('Num layers: %d' % len(model.layers))
            for l in range(len(model.layers)):
                print('Layer %d' % l)
                layer_function = K.function([model.layers[0].input], [model.layers[l].output])
                layer_output = layer_function([X])[0]
                Ys[l] = layer_output
            Y = np.zeros((X.shape[0], len(Ys)) + X.shape[1:])
            for l, values in Ys.items():
                Y[:, l, :] = values
            y = model.predict(X, *args, **kwargs)
            assert (Y[:, -1, :] == y).all()
            return Y

        model.predict = unfold_layer_outputs
        return model


class RnnModel(object):
    def __init__(self, feature_manager):
        self.feature_manager = feature_manager

    def get_feature_reshaper(self, mask_onset=None):
        if mask_onset is not None:
            mask_features = self.feature_manager.get_masked_features()
            return functools.partial(reshape_features_and_add_mask,
                                     mask_features=mask_features, mask_onset=mask_onset)
        else:
            return reshape_features

    def __call__(self, num_features):
        model = Sequential()
        model.add(SimpleURNN(name='RNN',
                             units=num_features, input_shape=(None, num_features),
                             activation='relu', inner_init='zero', return_sequences=True,
                             implementation=1  # consume less memory
                             ))
        return model


class ModelContainer:
    def __init__(self, X, Y, model_builder, row_provider, reshape_features_training, reshape_features_test):
        # assert Y.shape[1:] == X.shape[1:], \
        #     'feature sizes do not match: Y %s != X %s' % \
        #     (Y.shape[1:], X.shape[1:])
        self.original_shape = X.shape
        # self.X = self.flatten(X)
        # self.Y = self.flatten(Y)
        self.X, self.Y = X, Y
        self.row_provider = row_provider
        self.reshape_features_training = reshape_features_training
        self.reshape_features_prediction = reshape_features_test

        self.model = model_builder(self.X.shape[1:])
        self.model.compile(loss="mse", optimizer="rmsprop")

    def _train(self, model, train_indices, val_indices, num_epochs=10, batch_size=512, num_timesteps=5):
        best_model_path = 'best_model.k'
        save_best_model = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True)
        model.fit(self.reshape_features_training(self.X, train_indices, timesteps=num_timesteps),
                  self.reshape_features_training(self.Y, train_indices, timesteps=num_timesteps),
                  validation_data=(
                      self.reshape_features_training(self.X, val_indices, timesteps=num_timesteps),
                      self.reshape_features_training(self.Y, val_indices, timesteps=num_timesteps)),
                  batch_size=batch_size, epochs=num_epochs, verbose=1,
                  callbacks=[save_best_model])
        model.load_weights(best_model_path)

    def _predict(self, indices, timesteps=6, rnn=True, input_data=None):
        if not rnn and timesteps != 1:
            raise ValueError("not running an RNN model, so timesteps should be 1")

        X = self.reshape_features_prediction(self.X if input_data is None else input_data, indices, timesteps=timesteps)
        if TESTRUN:
            X = X[:5]
        predictions = self.model.predict(X)
        if not rnn:
            flat_predictions = np.zeros((predictions.shape[0], np.prod(predictions.shape[1:])))
            for i in range(predictions.shape[0]):
                flat_predictions[i] = predictions[i].flatten()
            return {0: flat_predictions}

        if predictions.shape[1] != timesteps:
            warnings.warn('predictions timesteps (%d) differ from requested timesteps (%d)'
                          % (predictions.shape[1], timesteps))
        Y = {}
        for t in range(predictions.shape[1]):
            Y[t] = predictions[:, t, :]
        return Y

    @staticmethod
    def _get_weights_file(kfold, suffix=''):
        weights_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'data', 'weights')
        if not os.path.isdir(weights_dir):
            os.mkdir(weights_dir)
        weights_filename = 'model_weights-kfold%d%s.hdf5' % (kfold, suffix)
        weights_file = os.path.join(weights_dir, weights_filename)
        return weights_file

    def cross_validate_prediction(self, train_epochs=10, max_timestep=6, weight_file_suffix='', rnn=True):
        """
        for each kfold, train on subset of features and predict the rest.
        Ultimately predict all features by concatenating them for each kfold.
        """
        initial_model_weights = self.model.get_weights()
        predicted_features = None
        num_kfold = 0
        for train_kfolds, val_kfolds, predict_kfolds in self.row_provider.get_kfolds():
            self.model.reset_states()
            self.model.set_weights(initial_model_weights)

            train_indices = self.row_provider.get_data_indices_from_kfolds(train_kfolds)
            validation_indices = self.row_provider.get_data_indices_from_kfolds(val_kfolds)
            predict_indices = self.row_provider.get_data_indices_from_kfolds(predict_kfolds)

            # TEST
            if TESTRUN:
                train_indices, validation_indices, predict_indices = \
                    train_indices[:3], validation_indices[:3], predict_indices[:3]

            weights_file = self._get_weights_file(num_kfold, weight_file_suffix)
            if os.path.isfile(weights_file):
                print('[kfold %d] using pre-trained weights %s' % (num_kfold, weights_file))
                self.model.load_weights(weights_file)
            elif train_epochs > 0:
                print('[kfold %d] training...' % num_kfold)
                self._train(self.model, train_indices, validation_indices,
                            num_epochs=train_epochs, num_timesteps=max_timestep)
                self.model.save_weights(weights_file, overwrite=True)
            print('[kfold %d] predicting...' % num_kfold)
            predicted_Y = self._predict(predict_indices, timesteps=max_timestep, rnn=rnn)
            for timestep, prediction in predicted_Y.items():
                if predicted_features is None:
                    predicted_features = np.zeros((len(predicted_Y),) + (self.X.shape[0],) + prediction.shape[1:])
                predicted_features[timestep, predict_indices, :] = prediction
            num_kfold += 1

        if rnn:
            predicted_features = np.reshape(predicted_features, ((len(predicted_features),) + self.original_shape))
        return predicted_features

    @staticmethod
    def flatten(m):
        return np.reshape(m, (m.shape[0], np.prod(m.shape[1:])))


def run_rnn():
    # params - fixed
    from model.feature_extractors.caffenet import CaffeNet
    models = {'caffenet': CaffeNet,
              'vgg16': functools.partial(KerasModel, VGG16, vgg16_preprocess),
              'vgg19': functools.partial(KerasModel, VGG19, vgg19_preprocess),
              'resnet50': functools.partial(KerasModel, ResNet50, resnet_preprocess),
              'resnet50-train': functools.partial(KerasTrainModel, ResNet50, resnet_preprocess),
              'inceptionv3': functools.partial(KerasModel, InceptionV3, inception_preprocess),
              'rnn': RnnModel,
              'feed-forward': FeedForwardModel}
    # params - command line
    parser = argparse.ArgumentParser(description='Train and predict whole features from occluded ones')
    parser.add_argument('--features_directory', type=str, default='~/group/features/',
                        help='directory containing features to load / target directory for predicted features')
    parser.add_argument('--input_features', type=str, default='train/alexnet-Wfc6.mat',
                        help='features to predict')
    parser.add_argument('--input_data_selection', type=int, nargs=2,
                        help='what part of the input data to use')
    parser.add_argument('--input_features_masked', type=str, default=None,
                        help='input features for mask onset')
    parser.add_argument('--target_features', type=str, default=None,
                        help="target output features, e.g. test/alexnet-fc7.mat'")
    parser.add_argument('--whole_features', type=str, default=None,
                        help='whole features (same as target for RNN, images for re-training)')
    parser.add_argument('--pres', type=str, default="../../../data/data_occlusion_klab325v2-pres.txt",
                        help='where to read the image ids from')
    parser.add_argument('--model', type=str, choices=models.keys(), default='rnn',
                        help='Model to train')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='how many epochs to search for optimal weights')
    parser.add_argument('--num_timesteps', type=int, default=6,
                        help='how many timesteps to run for prediction')
    parser.add_argument('--cross_validation', type=str, default='categories',
                        help='across what to validate')
    parser.add_argument('--add_whole_indices', action='store_true', default=False)
    parser.add_argument('--mask_onset', type=int, default=None,
                        help='at what timestep to use the masked input features')
    args = parser.parse_args()
    args.rnn = args.model in ['rnn', 'feed-forward']
    if args.rnn:
        if args.whole_features is not None and args.whole_features != args.target_features:
            raise ValueError('whole features should be equal to target features for RNNs')
        args.whole_features = args.target_features
    print('Running RNN with args', args)

    pres = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   args.pres), dtype='int')
    row_providers = {'images': RowsAcrossImages(),
                     'objects': RowsAcrossObjects(pres),
                     'categories': RowsAcrossCategories()}

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
    whole_features = feature_manager.load_features(args.whole_features)
    target_features = feature_manager.load_features(args.target_features)
    occluded_features = feature_manager.get_occluded_features(args.input_data_selection)
    print('loaded features with shape target=%s, occluded=%s, whole=%s' % (
        target_features.shape if target_features is not None else None,
        occluded_features.shape if occluded_features is not None else None,
        whole_features.shape if whole_features is not None else None))
    # assert whole_features.shape[1:] == occluded_features.shape[1:], \
    #     'feature sizes do not match: whole %s != occluded %s' % \
    #     (whole_features.shape[1:], occluded_features.shape[1:])
    if args.rnn:
        whole_features = target_features = align_features(target_features, occluded_features, pres)
    # TODO: conditionally offset with whole (np.concatenate(whole_features, occluded_features))
    if isinstance(row_provider, RowsAcrossImages):
        row_provider.set_num_images(occluded_features.shape[0])
    # model
    print('creating model...')
    model_builder = models[args.model](feature_manager=feature_manager)
    train_reshaper = model_builder.get_feature_reshaper(None)
    test_reshaper = model_builder.get_feature_reshaper(args.mask_onset)
    model = ModelContainer(occluded_features, target_features, model_builder, row_provider,
                           reshape_features_training=train_reshaper, reshape_features_test=test_reshaper)
    # run
    print('running...')
    predicted_Y = model.cross_validate_prediction(train_epochs=num_epochs, max_timestep=num_timesteps,
                                                  weight_file_suffix='-%s_%s' % (args.model, args.cross_validation),
                                                  rnn=args.rnn)
    if not args.rnn:
        whole_features = model._predict(list(range(len(whole_features) if not TESTRUN else 3)),
                                        input_data=whole_features, timesteps=num_timesteps, rnn=args.rnn)[0]
    # save
    file_suffix = '-%s_%s%s' % (args.model, args.cross_validation,
                                '-mask%d' % args.mask_onset if args.mask_onset is not None else '')
    filename, file_extension = os.path.splitext(args.target_features)
    save_basename = '%s%s%s' % (
        os.path.basename(filename) if args.rnn else '', file_suffix, "-rnn" if args.rnn else "")
    save_dir_predicted = os.path.dirname(args.input_features)
    save_dir_whole = os.path.dirname(args.target_features)
    print('saving to %s...' % save_basename)
    for timestep in range(len(predicted_Y)):
        features = predicted_Y[timestep]
        assert features.shape == occluded_features.shape if args.rnn \
            else features.shape[0] == occluded_features.shape[0], \
            "predicted features shape %s does not match input features shape %s" \
            % (features.shape, occluded_features.shape)
        timestep_basename = '%s-t%d%s' % (save_basename, timestep, file_extension)
        feature_manager.save(features, os.path.join(save_dir_predicted, timestep_basename))
        feature_manager.save(whole_features, os.path.join(save_dir_whole, timestep_basename))


if __name__ == '__main__':
    random.seed(0)
    run_rnn()
