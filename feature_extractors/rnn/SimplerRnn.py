import numpy as np
from keras import activations, initializers, regularizers
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import SimpleRNN, Recurrent


class SimplerRNN(SimpleRNN):
    def build(self, input_shape):
        super(SimplerRNN, self).build(input_shape)
        self.trainable_weights.remove(self.W)
        self.trainable_weights.remove(self.b)
        self.non_trainable_weights += [self.W, self.b]


class SimpleURNN(Recurrent):
    """Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - keras' SimpleRNN: code copied, removed all occurrences of W and b
    """

    def __init__(self, units,
                 inner_init='orthogonal',
                 activation='tanh',
                 recurrent_regularizer=None,
                 recurrent_dropout=0., **kwargs):
        self.units = units
        self.recurrent_initializer = initializers.get(inner_init)
        self.activation = activations.get(activation)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.recurrent_dropout = recurrent_dropout

        if self.recurrent_dropout:
            self.uses_learning_phase = True
        super(SimpleURNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.U = self.recurrent_initializer((self.units, self.units))  # , name='{}_U'.format(self.name))

        self.regularizers = []
        if self.recurrent_regularizer:
            self.recurrent_regularizer.set_param(self.U)
            self.regularizers.append(self.recurrent_regularizer)

        self.trainable_weights = [self.U]

        # if self.initial_weights is not None:
        #     self.set_weights(self.initial_weights)
        #     del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        output = self.activation(x + K.dot(prev_output * B_U, self.U))
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.units, 1)
            B_U = K.in_train_phase(K.dropout(ones, self.recurrent_dropout), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.units,
                  'inner_init': self.recurrent_initializer.__name__,
                  'activation': self.activation.__name__,
                  'U_regularizer': self.recurrent_regularizer.get_config() if self.recurrent_regularizer else None,
                  'dropout_U': self.recurrent_dropout}
        base_config = super(SimpleURNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
