# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import config as gf
import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, load_graph, letter_box_image

#FLAGS = tf.app.flags.FLAGS
#FLAGS= flags.FLAGS

gf.flags.DEFINE_string('input_img', '', 'Input image')
gf.flags.DEFINE_string('output_img', '', 'Output image')
gf.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
gf.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')
gf.flags.DEFINE_string('data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
gf.flags.DEFINE_string('ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
gf.flags.DEFINE_string('frozen_model', '', 'Frozen tensorflow protobuf model')
gf.flags.DEFINE_bool('tiny', False, 'Use tiny version of YOLOv3')

gf.flags.DEFINE_integer('size', 416, 'Image size')

gf.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
gf.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')

gf.flags.DEFINE_float('gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')


def main(argv=None):
    tf.compat.v1.disable_eager_execution()
    print(gf.FLAGS.gpu_memory_fraction)
    print(gf.FLAGS.input_img)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gf.FLAGS.gpu_memory_fraction)

    config = tf.compat.v1.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    img = Image.open(gf.FLAGS.input_img)
    img_resized = letter_box_image(img, gf.FLAGS.size, gf.FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names(gf.FLAGS.class_names)

    if gf.FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(gf.FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.compat.v1.Session(graph=frozenGraph, config=config) as sess:
            t0 = time.time()
            detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})

    else:
        if gf.FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), gf.FLAGS.size, gf.FLAGS.data_format)

        saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(scope='detector'))

        with tf.compat.v1.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, gf.FLAGS.ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()
            detected_boxes = sess.run(boxes, feed_dict={inputs: [img_resized]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    print("Predictions found in {:.2f}s".format(time.time() - t0))

    draw_boxes(filtered_boxes, img, classes, (gf.FLAGS.size, gf.FLAGS.size), True)

    img.save(gf.FLAGS.output_img)


if __name__ == '__main__':
    #tf.app.run()
    tf.compat.v1.app.run()