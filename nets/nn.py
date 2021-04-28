import math

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from utils import config

initializer = tf.random_normal_initializer(stddev=0.01)
l2 = tf.keras.regularizers.l2(4e-5)


def conv(x, filters, k=1, s=1):
    if s == 2:
        x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    else:
        padding = 'same'
    x = layers.Conv2D(filters, k, s, padding, use_bias=False,
                      kernel_initializer=initializer, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization(momentum=0.03)(x)
    x = layers.Activation(tf.nn.swish)(x)
    return x


def residual(x, filters, add=True):
    inputs = x
    if add:
        x = conv(x, filters, 1)
        x = conv(x, filters, 3)
        x = inputs + x
    else:
        x = conv(x, filters, 1)
        x = conv(x, filters, 3)
    return x


def csp(x, filters, n, add=True):
    y = conv(x, filters // 2)
    for _ in range(n):
        y = residual(y, filters // 2, add)

    x = conv(x, filters // 2)
    x = layers.concatenate([x, y])

    x = conv(x, filters)
    return x


def build_model(training=True):
    depth = config.depth[config.versions.index(config.version)]
    width = config.width[config.versions.index(config.version)]

    inputs = layers.Input([config.image_size, config.image_size, 3])
    x = tf.nn.space_to_depth(inputs, 2)
    x = conv(x, int(round(width * 64)), 3)
    x = conv(x, int(round(width * 128)), 3, 2)
    x = csp(x, int(round(width * 128)), int(round(depth * 3)))

    x = conv(x, int(round(width * 256)), 3, 2)
    x = csp(x, int(round(width * 256)), int(round(depth * 9)))
    x1 = x

    x = conv(x, int(round(width * 512)), 3, 2)
    x = csp(x, int(round(width * 512)), int(round(depth * 9)))
    x2 = x

    x = conv(x, int(round(width * 1024)), 3, 2)
    x = conv(x, int(round(width * 512)), 1, 1)
    x = layers.concatenate([x,
                            tf.nn.max_pool(x, 5,  1, 'SAME'),
                            tf.nn.max_pool(x, 9,  1, 'SAME'),
                            tf.nn.max_pool(x, 13, 1, 'SAME')])
    x = conv(x, int(round(width * 1024)), 1, 1)
    x = csp(x, int(round(width * 1024)), int(round(depth * 3)), False)

    x = conv(x, int(round(width * 512)), 1)
    x3 = x
    x = layers.UpSampling2D()(x)
    x = layers.concatenate([x, x2])
    x = csp(x, int(round(width * 512)), int(round(depth * 3)), False)

    x = conv(x, int(round(width * 256)), 1)
    x4 = x
    x = layers.UpSampling2D()(x)
    x = layers.concatenate([x, x1])
    x = csp(x, int(round(width * 256)), int(round(depth * 3)), False)
    p3 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name=f'p3_{len(config.class_dict)}',
                       kernel_initializer=initializer, kernel_regularizer=l2)(x)

    x = conv(x, int(round(width * 256)), 3, 2)
    x = layers.concatenate([x, x4])
    x = csp(x, int(round(width * 512)), int(round(depth * 3)), False)
    p4 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name=f'p4_{len(config.class_dict)}',
                       kernel_initializer=initializer, kernel_regularizer=l2)(x)

    x = conv(x, int(round(width * 512)), 3, 2)
    x = layers.concatenate([x, x3])
    x = csp(x, int(round(width * 1024)), int(round(depth * 3)), False)
    p5 = layers.Conv2D(3 * (len(config.class_dict) + 5), 1, name=f'p5_{len(config.class_dict)}',
                       kernel_initializer=initializer, kernel_regularizer=l2)(x)

    if training:
        return tf.keras.Model(inputs, [p5, p4, p3])
    else:
        return tf.keras.Model(inputs, Predict()([p5, p4, p3]))


def process_layer(feature_map, anchors):
    grid_size = tf.shape(feature_map)[1:3]
    ratio = tf.cast(tf.constant([config.image_size, config.image_size]) / grid_size, tf.float32)
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + len(config.class_dict)])

    box_centers, box_sizes, conf, prob = tf.split(feature_map, [2, 2, 1, len(config.class_dict)], axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)

    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * ratio[::-1]

    box_sizes = tf.exp(box_sizes) * rescaled_anchors
    box_sizes = box_sizes * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    return x_y_offset, boxes, conf, prob


def box_iou(pred_boxes, valid_true_boxes):
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    intersect_min = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_max = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)

    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    return intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)


def compute_nms(args):
    boxes, classification = args

    def nms_fn(score, label):
        score_indices = tf.where(backend.greater(score, config.threshold))

        filtered_boxes = tf.gather_nd(boxes, score_indices)
        filtered_scores = backend.gather(score, score_indices)[:, 0]

        nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, config.max_boxes, 0.1)
        score_indices = backend.gather(score_indices, nms_indices)

        label = tf.gather_nd(label, score_indices)
        score_indices = backend.stack([score_indices[:, 0], label], axis=1)

        return score_indices

    all_indices = []
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * tf.ones((backend.shape(scores)[0],), dtype='int64')
        all_indices.append(nms_fn(scores, labels))
    indices = backend.concatenate(all_indices, axis=0)

    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=backend.minimum(config.max_boxes, backend.shape(scores)[0]))

    indices = backend.gather(indices[:, 0], top_indices)
    boxes = backend.gather(boxes, indices)
    labels = backend.gather(labels, top_indices)

    pad_size = backend.maximum(0, config.max_boxes - backend.shape(scores)[0])

    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = backend.cast(labels, 'int32')

    boxes.set_shape([config.max_boxes, 4])
    scores.set_shape([config.max_boxes])
    labels.set_shape([config.max_boxes])

    return [boxes, scores, labels]


class ComputeLoss(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_loss(y_pred, y_true, anchors):
        grid_size = tf.shape(y_pred)[1:3]
        ratio = tf.cast(tf.constant([config.image_size, config.image_size]) / grid_size, tf.float32)
        batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf, pred_prob = process_layer(y_pred, anchors)

        object_mask = y_true[..., 4:5]

        def cond(idx, _):
            return tf.less(idx, tf.cast(batch_size, tf.int32))

        def body(idx, mask):
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4],
                                               tf.cast(object_mask[idx, ..., 0], 'bool'))
            iou = box_iou(pred_boxes[idx], valid_true_boxes)
            return idx + 1, mask.write(idx, tf.cast(tf.reduce_max(iou, axis=-1) < 0.2, tf.float32))

        ignore_mask = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        _, ignore_mask = tf.while_loop(cond=cond, body=body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_boxes[..., 0:2] / ratio[::-1] - x_y_offset

        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_boxes[..., 2:4] / anchors
        true_tw_th = tf.where(tf.equal(true_tw_th, 0), tf.ones_like(true_tw_th), true_tw_th)
        pred_tw_th = tf.where(tf.equal(pred_tw_th, 0), tf.ones_like(pred_tw_th), pred_tw_th)
        true_tw_th = tf.math.log(tf.clip_by_value(true_tw_th, 1e-9, 1e+9))
        pred_tw_th = tf.math.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e+9))

        box_loss_scale = y_true[..., 2:3] * y_true[..., 3:4]
        box_loss_scale = 2. - box_loss_scale / tf.cast(config.image_size ** 2, tf.float32)

        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale)
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale)

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf)

        conf_loss = tf.reduce_sum((conf_loss_pos + conf_loss_neg))

        true_conf = y_true[..., 5:]

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_conf, pred_prob)
        class_loss = tf.reduce_sum(class_loss)

        return xy_loss + wh_loss + conf_loss + class_loss

    def __call__(self, y_pred, y_true):
        loss = 0.
        anchor_group = [config.anchors[6:9], config.anchors[3:6], config.anchors[0:3]]

        for i in range(len(y_pred)):
            loss += self.compute_loss(y_pred[i], y_true[i], anchor_group[i])
        return loss


class CosineLR(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, steps):
        super().__init__()
        self.lr = 0.008 * config.batch_size / 64
        self.warmup_init = 0.0008
        self.warmup_step = steps
        self.decay_steps = tf.cast((config.num_epochs - 1) * self.warmup_step, tf.float32)

    def __call__(self, step):
        linear_warmup = tf.cast(step, dtype=tf.float32) / self.warmup_step * (self.lr - self.warmup_init)
        cosine_lr = 0.5 * self.lr * (1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
        return tf.where(step < self.warmup_step, self.warmup_init + linear_warmup, cosine_lr)

    def get_config(self):
        pass


class Predict(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        y_pred = [(inputs[0], config.anchors[6:9]),
                  (inputs[1], config.anchors[3:6]),
                  (inputs[2], config.anchors[0:3])]

        boxes_list, conf_list, prob_list = [], [], []
        for result in [process_layer(feature_map, anchors) for (feature_map, anchors) in y_pred]:
            x_y_offset, box, conf, prob = result
            grid_size = tf.shape(x_y_offset)[:2]
            box = tf.reshape(box, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf = tf.reshape(conf, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob = tf.reshape(prob, [-1, grid_size[0] * grid_size[1] * 3, len(config.class_dict)])
            boxes_list.append(box)
            conf_list.append(tf.sigmoid(conf))
            prob_list.append(tf.sigmoid(prob))

        boxes = tf.concat(boxes_list, axis=1)
        conf = tf.concat(conf_list, axis=1)
        prob = tf.concat(prob_list, axis=1)

        center_x, center_y, w, h = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        x_max = center_x + w / 2
        y_max = center_y + h / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        outputs = tf.map_fn(fn=compute_nms,
                            elems=[boxes, conf * prob],
                            dtype=['float32', 'float32', 'int32'],
                            parallel_iterations=100)

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], config.max_boxes, 4),
                (input_shape[1][0], config.max_boxes),
                (input_shape[1][0], config.max_boxes), ]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs) + 1) * [None]

    def get_config(self):
        return super().get_config()
