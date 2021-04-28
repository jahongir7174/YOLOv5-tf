import argparse
import multiprocessing
import os
import sys

import cv2
import numpy
import tensorflow as tf
import tqdm

from nets import nn
from utils import config
from utils import util
from utils.dataset import input_fn, DataLoader

numpy.random.seed(12345)
tf.random.set_seed(12345)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train():
    strategy = tf.distribute.MirroredStrategy()

    file_names = []
    with open(os.path.join(config.data_dir, 'train.txt')) as f:
        for file_name in f.readlines():
            image_path = os.path.join(config.data_dir, config.image_dir, file_name.rstrip() + '.jpg')
            label_path = os.path.join(config.data_dir, config.label_dir, file_name.rstrip() + '.xml')
            if os.path.exists(image_path) and os.path.exists(label_path):
                if os.path.exists(os.path.join(config.data_dir, 'TF')):
                    file_names.append(os.path.join(config.data_dir, 'TF', file_name.rstrip() + '.tf'))
                else:
                    file_names.append(file_name.rstrip())

    steps = len(file_names) // config.batch_size
    if os.path.exists(os.path.join(config.data_dir, 'TF')):
        dataset = DataLoader().input_fn(file_names)
    else:
        dataset = input_fn(file_names)
    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        model = nn.build_model()
        model.summary()
        optimizer = tf.keras.optimizers.Adam(nn.CosineLR(steps), 0.937)

    with strategy.scope():
        loss_object = nn.ComputeLoss()

        def compute_loss(y_true, y_pred):
            total_loss = loss_object(y_pred, y_true)
            return tf.reduce_sum(total_loss) / config.batch_size

    with strategy.scope():
        def train_step(image, y_true):
            with tf.GradientTape() as tape:
                y_pred = model(image, training=True)
                loss = compute_loss(y_true, y_pred)
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return loss

    with strategy.scope():
        @tf.function
        def distributed_train_step(image, y_true):
            per_replica_losses = strategy.run(train_step, args=(image, y_true))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train_fn():
        if not os.path.exists('weights'):
            os.makedirs('weights')
        pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
        print(f'[INFO] {len(file_names)} data points')
        for step, inputs in enumerate(dataset):
            if step % steps == 0:
                print(f'Epoch {step // steps + 1}/{config.num_epochs}')
                pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
            step += 1
            image, y_true_1, y_true_2, y_true_3 = inputs
            y_true = (y_true_1, y_true_2, y_true_3)
            loss = distributed_train_step(image, y_true)
            pb.add(1, [('loss', loss)])
            if step % steps == 0:
                model.save_weights(os.path.join("weights", f"model_{config.version}.h5"))
            if step // steps == config.num_epochs:
                sys.exit("--- Stop Training ---")

    train_fn()


def test():
    def draw_bbox(image, boxes):
        for box in boxes:
            coordinate = numpy.array(box[:4], dtype=numpy.int32)
            c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
            cv2.rectangle(image, c1, c2, (255, 0, 0), 1)
        return image

    def test_fn():
        if not os.path.exists('results'):
            os.makedirs('results')
        file_names = []
        with open(os.path.join(config.data_dir, 'test.txt')) as f:
            for file_name in f.readlines():
                file_names.append(file_name.rstrip())

        model = nn.build_model(training=False)
        model.load_weights(f"weights/model_{config.version}.h5", True)

        for file_name in tqdm.tqdm(file_names):
            image = cv2.imread(os.path.join(config.data_dir, config.image_dir, file_name + '.jpg'))
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_np, scale, dw, dh = util.resize(image_np)
            image_np = image_np.astype(numpy.float32) / 255.0

            boxes, scores, labels = model.predict(image_np[numpy.newaxis, ...])

            boxes, scores, labels = numpy.squeeze(boxes, 0), numpy.squeeze(scores, 0), numpy.squeeze(labels, 0)

            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
            image = draw_bbox(image, boxes)
            cv2.imwrite(f'results/{file_name}.jpg', image)

    test_fn()


def write_tf_record(queue, sentinel):
    def byte_feature(value):
        if not isinstance(value, bytes):
            if not isinstance(value, list):
                value = value.encode('utf-8')
            else:
                value = [val.encode('utf-8') for val in value]
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    while True:
        file_name = queue.get()

        if file_name == sentinel:
            break
        in_image = util.load_image(file_name)[:, :, ::-1]
        boxes, label = util.load_label(file_name)

        in_image, boxes = util.resize(in_image, boxes)

        y_true_1, y_true_2, y_true_3 = util.process_box(boxes, label)

        in_image = in_image.astype('float32')
        y_true_1 = y_true_1.astype('float32')
        y_true_2 = y_true_2.astype('float32')
        y_true_3 = y_true_3.astype('float32')

        in_image = in_image.tobytes()
        y_true_1 = y_true_1.tobytes()
        y_true_2 = y_true_2.tobytes()
        y_true_3 = y_true_3.tobytes()

        features = tf.train.Features(feature={'in_image': byte_feature(in_image),
                                              'y_true_1': byte_feature(y_true_1),
                                              'y_true_2': byte_feature(y_true_2),
                                              'y_true_3': byte_feature(y_true_3)})
        tf_example = tf.train.Example(features=features)
        opt = tf.io.TFRecordOptions('GZIP')
        with tf.io.TFRecordWriter(os.path.join(config.data_dir, 'TF', file_name + ".tf"), opt) as writer:
            writer.write(tf_example.SerializeToString())


def generate_tf_record():
    if not os.path.exists(os.path.join(config.data_dir, 'TF')):
        os.makedirs(os.path.join(config.data_dir, 'TF'))
    file_names = []
    with open(os.path.join(config.data_dir, 'train.txt')) as reader:
        for line in reader.readlines():
            file_names.append(line.rstrip().split(' ')[0])
    sentinel = ("", [])
    queue = multiprocessing.Manager().Queue()
    for file_name in tqdm.tqdm(file_names):
        queue.put(file_name)
    for _ in range(os.cpu_count()):
        queue.put(sentinel)
    print('[INFO] generating TF record')
    process_pool = []
    for i in range(os.cpu_count()):
        process = multiprocessing.Process(target=write_tf_record, args=(queue, sentinel))
        process_pool.append(process)
        process.start()
    for process in process_pool:
        process.join()


class AnchorGenerator:
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.num_cluster

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = numpy.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = numpy.tile(cluster_area, [1, n])
        cluster_area = numpy.reshape(cluster_area, (n, k))

        box_w_matrix = numpy.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = numpy.reshape(numpy.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = numpy.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = numpy.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = numpy.reshape(numpy.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = numpy.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = numpy.multiply(min_w_matrix, min_h_matrix)

        return inter_area / (box_area + cluster_area - inter_area)

    def avg_iou(self, boxes, clusters):
        accuracy = numpy.mean([numpy.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def generator(self, boxes, k, dist=numpy.median):
        box_number = boxes.shape[0]
        last_nearest = numpy.zeros((box_number,))
        clusters = boxes[numpy.random.choice(box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self.iou(boxes, clusters)

            current_nearest = numpy.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest

        return clusters

    def generate_anchor(self):
        boxes = self.get_boxes()
        result = self.generator(boxes, k=self.num_cluster)
        result = result[numpy.lexsort(result.T[0, None])]
        print("\nAnchors: \n{}".format(result))
        print("\nFitness: {:.4f}".format(self.avg_iou(boxes, result)))

    @staticmethod
    def get_boxes():
        boxes = []
        file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.label_dir))]
        for file_name in file_names:
            for box in util.load_label(file_name)[0]:
                boxes.append([box[2] - box[0], box[3] - box[1]])
        return numpy.array(boxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    if args.anchor:
        AnchorGenerator(9).generate_anchor()
    if args.record:
        generate_tf_record()
    if args.train:
        train()
    if args.test:
        test()
