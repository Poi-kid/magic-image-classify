import os
import numpy as np
import tensorflow as tf
import input_data
import model
from PIL import Image


N_CLASSES = 6
IMG_W = 128
IMG_H = 96
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 20000
learning_rate = 0.0001
CHANNELS = 1


def run_training():
    train_dir = 'data\\'
    logs_train_dir = 'model\\'
    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    train_logits = model.inference(train_batch, N_CLASSES, CHANNELS)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 5 == 0:
                print('Step %d, train loss = %f, train accuracy = %f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 500 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def evaluate_image_from_client(func, arg1):
    max_index = -1
    logs_train_dir = 'model\\'

    x = tf.placeholder(tf.float32, shape=(1, IMG_H, IMG_W, CHANNELS))
    logit = model.inference(x, N_CLASSES, CHANNELS)
    logit = tf.nn.softmax(logit)

    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')

        img = tf.random_normal((1, IMG_H, IMG_W, CHANNELS)).eval()
        sess.run(logit, feed_dict={x: img})

        while True:
            close = func(arg1, max_index)
            if close:
                arg1.close()
                break
            log = "save.png"
            img = Image.open(log).convert('L')
            img = np.array(img, 'f').reshape((1, IMG_H, IMG_W, CHANNELS))
            prediction = sess.run(logit, feed_dict={x: img})
            max_index = np.argmax(prediction)
            print(max_index, prediction)
            
            
if __name__ == "__main__":
    run_training()


# def test():
#     delta = 0
#     n = 0
#     x = tf.placeholder(tf.float32, shape=(1, IMG_H, IMG_W, CHANNELS))
#     logit = model.inference(x, N_CLASSES, CHANNELS)
#     logit = tf.nn.softmax(logit)
#     logs_train_dir = '2\\'
#     saver = tf.train.Saver()
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#         ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             print('No checkpoint file found')
#         for i in range(50):
#             for j in range(6):
#                 try:
#                     old = time.time()
#                     log = "data\\" + str(j) + "--" + str(i) + ".png"
#                     img = Image.open(log).convert('L')
#                     img = np.array(img, 'f').reshape((1, IMG_H, IMG_W, CHANNELS))
#                     prediction = sess.run(logit, feed_dict={x: img})
#                     max_index = np.argmax(prediction)
#                     print(max_index, prediction, (time.time() - old))
#                 except:
#                     pass
#
#
# test()
