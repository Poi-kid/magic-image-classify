import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def get_files(path):
    image_list = []
    label_list = []

    # 我已经将剪切好的数字图片进行人工分类
    # 并且在图片名称里标明了它们的标签
    # 例如0--1.jpg, 0--2.jpg, 0--3.jpg
    # 所以用字符'--'去切割就能得到标签0, 1, 2, 3...
    for f in os.listdir(path):
        image_list.append(path+'\\'+f)
        label_list.append(int(f.split("--")[0]))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list


def get_batch(image, label, image_w, image_h, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    image = input_queue[0]
    label = input_queue[1]
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_image_with_crop_or_pad(image, image_h, image_w)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch