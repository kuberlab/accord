import tensorflow as tf
from accord.unet import unet
from kibernetika.rpt import MlBoardReporter
import glob
import os
import cv2
import voc.utils as utils
import numpy as np


def data_fn(params, training):
    examples = []
    for f in glob.glob(params['data_set'] + '/*.xml'):
        name = os.path.basename(f)
        name = name.replace('.xml', '.png')
        img = cv2.imread(os.path.join(params['data_set'], name), cv2.IMREAD_COLOR)
        boxes = utils.generate(f)
        label = utils.gen_mask((img.shape[0], img.shape[1]), boxes)
        examples.append((img, label))
    r = params['resolution']

    def _input_fn():
        def _generator():
            for img, label in examples:
                img = cv2.resize(img, (r, r), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (r, r), interpolation=cv2.INTER_LINEAR)
                img = img.astype(np.float32)/255.0
                label = label.astype(np.float32)/255.0
                yield img, label

        ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                            (tf.TensorShape([r, r, 3]), tf.TensorShape([r, r, len(utils.clazzes)])))

        if training:
            ds = ds.shuffle(params['batch_size'] * 2, reshuffle_each_iteration=True)
        if training:
            ds = ds.repeat(params['num_epochs']).prefetch(params['batch_size'] * 2)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        return ds

    return _input_fn


def _unet_model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    resolution = params['resolution']
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['image']
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    outs = unet(features, len(utils.clazzes), params['num_chans'], params['drop_prob'],
                params['num_pools'], training=training)
    outs = tf.nn.softmax(outs)
    loss = None
    train_op = None
    hooks = []
    export_outputs = None
    eval_hooks = []
    chief_hooks = []
    metrics = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.absolute_difference(outs, labels)
        global_step = tf.train.get_or_create_global_step()
        if training:
            board_hook = MlBoardReporter({
                "_step": global_step,
                "_train_loss": loss}, every_steps=params['save_summary_steps'])
            chief_hooks = [board_hook]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if params['optimizer'] == 'AdamOptimizer':
                    opt = tf.train.AdamOptimizer(float(params['lr']))
                else:
                    opt = tf.train.RMSPropOptimizer(float(params['lr']), params['weight_decay'])
                train_op = opt.minimize(loss, global_step=global_step)

        tf.summary.image('Reconstruction_Date', tf.reshape(outs[:, :, :, utils.clazzes['Date']],
                                                           [params['batch_size'], resolution, resolution, 1]), 3)
        tf.summary.image('Original_Data', tf.reshape(labels[:, :, :, utils.clazzes['Date']],
                                                     [params['batch_size'], resolution, resolution, 1]), 3)
        tf.summary.image('Reconstruction_Producer', tf.reshape(outs[:, :, :, utils.clazzes['Producer']],
                                                               [params['batch_size'], resolution, resolution, 1]), 3)
        tf.summary.image('Original_Producer', tf.reshape(labels[:, :, :, utils.clazzes['Producer']],
                                                         [params['batch_size'], resolution, resolution, 1]), 3)
        tf.summary.image('Reconstruction_Limits2Column', tf.reshape(outs[:, :, :, utils.clazzes['Limits2Column']],
                                                                    [params['batch_size'], resolution, resolution, 1]),
                         3)
        tf.summary.image('Original_Limits2Column', tf.reshape(labels[:, :, :, utils.clazzes['Limits2Column']],
                                                              [params['batch_size'], resolution, resolution, 1]), 3)
        hooks = []

    else:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                {'mask': outs})}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions={'mask': outs},
        training_chief_hooks=chief_hooks,
        loss=loss,
        training_hooks=hooks,
        export_outputs=export_outputs,
        evaluation_hooks=eval_hooks,
        train_op=train_op)


class BoxUnet(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _unet_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir,
            )

        super(BoxUnet, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
