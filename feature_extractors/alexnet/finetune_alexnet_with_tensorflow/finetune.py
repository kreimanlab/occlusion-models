"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import argparse
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Iterator

from model.feature_extractors.alexnet.finetune_alexnet_with_tensorflow.alexnet import AlexNet
from model.feature_extractors.alexnet.finetune_alexnet_with_tensorflow.datagenerator import ImageDataGenerator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Finetune Alexnet')
    parser.add_argument('--kfold', type=int, default=1)
    parser.add_argument('--images_dirname', type=str, default='images')
    parser.add_argument('--from_checkpoint', action='store_true', default=False)
    parser.add_argument('--no-from_checkpoint', action='store_false', dest='from_checkpoint')
    parser.add_argument('--prototype_run', action='store_true', default=False)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=20, help='how often to write the tf.summary to disk')
    parser.add_argument('--loss', type=str, default='features')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)  # TODO: make sure this still trains well (was 128)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info('Running with args {:s}'.format(', '.join(
        ['{}={}'.format(key, value) for (key, value) in vars(args).items()])))

    """
    Configuration Part.
    """
    # Path to the textfiles for the trainings and validation set
    data_path = os.path.join(os.path.dirname(__file__),
                             args.images_dirname + ('' if not args.prototype_run else '.proto'))
    train_file = os.path.join(data_path, 'train{}.txt'.format(args.kfold))
    val_file = os.path.join(data_path, 'val{}.txt'.format(args.kfold))
    test_file = os.path.join(data_path, 'test{}.txt'.format(args.kfold))
    predictions_file = os.path.join(data_path, "predictions{}.txt".format(args.kfold))

    # Network params
    skip_layers = []  # train everything, not just ['fc8', 'fc7', 'fc6']
    num_classes = 325 if args.loss == 'categorical' else None
    logger.debug("Num classes: {}".format(num_classes))

    # Path for tf.summary.FileWriter and to store model checkpoints
    storage_path = os.path.join(os.path.dirname(__file__), "storage", "kfold{}".format(args.kfold))
    filewriter_path = os.path.join(storage_path, "tensorboard")
    checkpoint_path = os.path.join(storage_path, "checkpoints")

    """
    Main Part of the finetuning Script.
    """

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        logger.info('Loading data')
        tr_data = ImageDataGenerator(train_file,
                                     mode='training',
                                     num_classes=num_classes,
                                     batch_size=args.batch_size,
                                     shuffle=True)
        val_data = ImageDataGenerator(val_file,
                                      mode='inference',
                                      num_classes=num_classes,
                                      batch_size=args.batch_size,
                                      shuffle=False)
        test_data = ImageDataGenerator(test_file,
                                       mode='inference',
                                       num_classes=num_classes,
                                       batch_size=args.batch_size,
                                       shuffle=False)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)
    test_init_op = iterator.make_initializer(test_data.data)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [args.batch_size, 227, 227, 3], name='x')
    y = tf.placeholder(tf.float32, [args.batch_size, 4096 if args.loss == 'features' else num_classes], name='y')
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    logger.info('Creating model')
    model = AlexNet(x, keep_prob, skip_layer=skip_layers + ['fc8'] if args.loss == 'categorical' else [],
                    num_classes=num_classes)

    # Link variable to model output
    class_output = model.fc8
    features_output = model.fc7

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if not skip_layers or v.name.split('/')[0] in skip_layers]

    # Op for calculating the loss
    with tf.name_scope("loss"):
        if args.loss == 'features':
            loss = tf.losses.mean_squared_error(predictions=features_output, labels=y)
        elif args.loss == 'categorical':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=class_output, labels=y))
            # loss = -tf.reduce_sum(y * tf.log(class_output + 1e-10))
        else:
            raise ValueError('Invalid value for loss provided: {}'.format(args.loss))

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
        gradients = [(g, v) for g, v in gradients if g is not None]

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('loss', loss)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = max(int(np.floor(tr_data.data_size / args.batch_size)), 1)
    val_batches_per_epoch = max(int(np.floor(val_data.data_size / args.batch_size)), 1)
    test_batches_per_epoch = max(int(np.floor(test_data.data_size / args.batch_size)), 1)

    # Start Tensorflow session
    with tf.Session() as sess:
        start_epoch = 0
        if args.from_checkpoint:
            checkpoints = [f[:-6] for f in
                           [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path)
                            if f.endswith(".ckpt.index")] if os.path.isfile(f)]
            if len(checkpoints) == 0:
                logger.warning("No checkpoints found - starting from scratch")
                args.from_checkpoint = False
            else:
                last_checkpoint = sorted(checkpoints)[-1]
                try:
                    saver.restore(sess, last_checkpoint)
                    start_epoch = int(
                        last_checkpoint[-6]) + 1  # TODO: fix hard-coded single-digit (doesn't work for 11)
                except Exception:
                    logger.warning("Unable to recover checkpoint {} - starting from scratch".format(last_checkpoint))
                    args.from_checkpoint = False
        if not args.from_checkpoint:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(sess)

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        logger.info("{} Start training...".format(datetime.now()))
        logger.info("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

        # Loop over number of epochs
        best_val_epoch, best_val_loss = None, math.inf
        for epoch in range(start_epoch, args.num_epochs):

            logger.info("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)

            for step in range(train_batches_per_epoch):
                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)

                # And run the training op
                sess.run(train_op, feed_dict={x: img_batch,
                                              y: label_batch,
                                              keep_prob: args.dropout_rate})

                # Generate summary with the current batch of data and write to file
                if True or step % args.display_step == 0:  # TODO: remove True
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})

                    writer.add_summary(s, epoch * train_batches_per_epoch + step)

            # save checkpoint of the model
            if epoch % args.save_every == 0:
                logger.info("{} Saving checkpoint of model...".format(datetime.now()))
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch{:d}.ckpt'.format(epoch + 1))
                save_path = saver.save(sess, checkpoint_name)
                logger.info("{} Model checkpoint saved at {}".format(datetime.now(), save_path))

            # Validate the model on the entire validation set
            logger.info("{} Start validation".format(datetime.now()))
            sess.run(validation_init_op)
            val_loss = 0.
            val_count = 0
            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                _loss = sess.run(loss, feed_dict={x: img_batch,
                                                  y: label_batch,
                                                  keep_prob: 1.})
                val_loss += _loss
                val_count += 1
            val_loss /= val_count
            logger.info("{} Validation Loss = {:.4f}".format(datetime.now(), val_loss))
            if val_loss < best_val_loss:
                best_val_loss, best_val_epoch = val_loss, epoch
                saver.save(sess, os.path.join(checkpoint_path, 'model_best.ckpt'))

                # Test the model
                logger.info("{} Start test".format(datetime.now()))
                test_predictions = np.empty((test_data.data_size, num_classes or 4096))
                sess.run(test_init_op)
                for batch_num in range(test_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    preds = sess.run(features_output, feed_dict={x: img_batch, keep_prob: 1.})
                    test_predictions[batch_num * args.batch_size:(batch_num + 1) * args.batch_size, :] = preds
                with open(predictions_file, 'w') as f:
                    for img_path, prediction in zip(test_data._img_paths, test_predictions):
                        f.write('{} {}\n'.format(img_path, ",".join([str(p) for p in prediction])))
                logger.info("Wrote predictions to {}".format(predictions_file))
            elif epoch - best_val_epoch > args.patience:
                logger.info("Validation loss has not decreased for {:d} epochs - stop (best epoch: {:d})".format(
                    epoch + 1 - best_val_epoch, best_val_epoch))
                break


if __name__ == '__main__':
    main()
