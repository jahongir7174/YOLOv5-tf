import os
import numpy
import tensorflow as tf
from tensorflow.keras import utils

from utils import config, util


class Generator(utils.Sequence):
    def __init__(self, file_names):
        self.file_names = file_names

    def __len__(self):
        return int(numpy.floor(len(self.file_names) / config.batch_size))

    def __getitem__(self, index):
        image = util.load_image(self.file_names[index])
        boxes, label = util.load_label(self.file_names[index])
        image, boxes = util.resize(image, boxes)
        image, boxes = util.random_flip(image, boxes)

        image = image[:, :, ::-1].astype(numpy.float32)
        image = image / 255.0
        y_true_1, y_true_2, y_true_3 = util.process_box(boxes, label)
        return image, y_true_1, y_true_2, y_true_3

    def on_epoch_end(self):
        numpy.random.shuffle(self.file_names)


def input_fn(file_names):
    def generator_fn():
        generator = utils.OrderedEnqueuer(Generator(file_names), True)
        generator.start(workers=min(os.cpu_count(), config.batch_size))
        while True:
            image, y_true_1, y_true_2, y_true_3 = generator.get().__next__()
            yield image, y_true_1, y_true_2, y_true_3

    output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
    output_shapes = ((config.image_size, config.image_size, 3),
                     (config.image_size // 32, config.image_size // 32, 3, len(config.class_dict) + 5),
                     (config.image_size // 16, config.image_size // 16, 3, len(config.class_dict) + 5),
                     (config.image_size // 8,  config.image_size // 8,  3, len(config.class_dict) + 5))

    dataset = tf.data.Dataset.from_generator(generator=generator_fn,
                                             output_types=output_types,
                                             output_shapes=output_shapes)

    dataset = dataset.repeat(config.num_epochs + 1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class DataLoader:
    def __init__(self):
        super().__init__()
        self.description = {'in_image': tf.io.FixedLenFeature([], tf.string),
                            'y_true_1': tf.io.FixedLenFeature([], tf.string),
                            'y_true_2': tf.io.FixedLenFeature([], tf.string),
                            'y_true_3': tf.io.FixedLenFeature([], tf.string)}

    def parse_data(self, tf_record):
        features = tf.io.parse_single_example(tf_record, self.description)

        in_image = tf.io.decode_raw(features['in_image'], tf.float32)
        in_image = tf.reshape(in_image, (config.image_size, config.image_size, 3))
        in_image = in_image / 255.

        y_true_1 = tf.io.decode_raw(features['y_true_1'], tf.float32)
        y_true_1 = tf.reshape(y_true_1,
                              (config.image_size // 32, config.image_size // 32, 3, 5 + len(config.class_dict)))

        y_true_2 = tf.io.decode_raw(features['y_true_2'], tf.float32)
        y_true_2 = tf.reshape(y_true_2,
                              (config.image_size // 16, config.image_size // 16, 3, 5 + len(config.class_dict)))

        y_true_3 = tf.io.decode_raw(features['y_true_3'], tf.float32)
        y_true_3 = tf.reshape(y_true_3,
                              (config.image_size // 8,  config.image_size // 8,  3, 5 + len(config.class_dict)))

        return in_image, y_true_1, y_true_2, y_true_3

    def input_fn(self, file_names):
        dataset = tf.data.TFRecordDataset(file_names, 'GZIP')
        dataset = dataset.map(self.parse_data, os.cpu_count())
        dataset = dataset.repeat(config.num_epochs + 1)
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
