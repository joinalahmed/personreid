'''
Script to test traffic light localization and detection
'''

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import visualization_utils


class PersonDetector(object):
    def __init__(self):

        self.car_boxes = []
        PATH_TO_CKPT = '/home/leadics-18-2/Desktop/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    def get_localization(self, image, visual=False):
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """
        category_index = {1: {'id': 1, 'name': u'person'},
                          2: {'id': 2, 'name': u'bicycle'},
                          3: {'id': 3, 'name': u'car'},
                          4: {'id': 4, 'name': u'motorcycle'},
                          5: {'id': 5, 'name': u'airplane'},
                          6: {'id': 6, 'name': u'bus'},
                          7: {'id': 7, 'name': u'train'},
                          8: {'id': 8, 'name': u'truck'},
                          9: {'id': 9, 'name': u'boat'},
                          10: {'id': 10, 'name': u'traffic light'},
                          11: {'id': 11, 'name': u'fire hydrant'},
                          13: {'id': 13, 'name': u'stop sign'},
                          14: {'id': 14, 'name': u'parking meter'}}

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            if visual == True:
                visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True, min_score_thresh=.4,
                    line_thickness=3)

                plt.figure(figsize=(9, 6))
                plt.imshow(image)
                plt.show()

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            cls = classes.tolist()

            # The ID for car is 3
            idx_vec = [i for i, v in enumerate(cls) if ((v == 1) and (scores[i] > 0.7))]

            if len(idx_vec) == 0:
                print('no detection!')
                tmp_car_boxes = []
                self.car_boxes = tmp_car_boxes
            else:
                tmp_car_boxes = []
                for idx in idx_vec:
                    dim = image.shape[0:2]
                    box = self.box_normal_to_pixel(boxes[idx], dim)
                    box_h = box[2] - box[0]
                    box_w = box[3] - box[1]
                    ratio = box_h / (box_w + 0.01)

                    # if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                    tmp_car_boxes.append(box)
                    print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                    '''   
                      else:
                          print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)
                      '''

                self.car_boxes = tmp_car_boxes

        return self.car_boxes
