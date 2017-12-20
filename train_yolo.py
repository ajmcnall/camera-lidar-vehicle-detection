"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default='./Car_detection_data/new_car_images/')

argparser.add_argument(
    '--class_num', dest='class_num',
    help='path to classes file',
    default=23,
    type=int)
# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    class_num = args.class_num


    anchors = YOLO_ANCHORS

    model_body, model = create_model(anchors, class_num)    
    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    train(model, class_num, anchors, data_path, epochs=5, batch_size=32, \
        save_weights='trained_stage_1.h5', earlystop=False)
    
    model_body, model = create_model(anchors, class_num, load_pretrained=False, freeze_body=False)
    model.load_weights('trained_stage_1.h5')
    model.compile(optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})
    train(model, class_num, anchors, data_path, epochs=30, batch_size=8, \
        save_weights='trained_stage_2.h5', earlystop=True)



def load_img_box(path, idx):  
    images = []
    boxes = []
    for i in idx:
        image_path = os.path.join(path, str(i) + '_image.jpg')
        box_path = os.path.join(path, str(i) + '_image.txt')
        img = PIL.Image.open(image_path)
        images.append(img)
        box = np.loadtxt(box_path)
        boxes.append(box)
    return images, boxes


def process_data(images, boxes):
    '''processes the data'''
    # images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)
    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    boxes = [box.reshape((-1, 5)) for box in boxes]
    boxes_xy = [box[:, 1:3] for box in boxes]
    boxes_wh = [box[:, 3:] for box in boxes]
    boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
    boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
    boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

    # find the max number of boxes
    max_boxes = 24
    # add zero pad for training
    for i, boxz in enumerate(boxes):
        if boxz.shape[0]  < max_boxes:
            zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
            boxes[i] = np.vstack((boxz, zero_padding))

    return np.array(processed_images)[:,:,:,3:], np.array(boxes)


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_num, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), class_num)
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    top10_yolo = Model(yolo_model.input, yolo_model.layers[10].output)
    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+class_num), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': class_num})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return yolo_model, model

def train(model, class_num, anchors, data_path, epochs, batch_size, save_weights, earlystop):

    # logging = TensorBoard()

    # if earlystop:
    #     checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
    #                              save_weights_only=True, save_best_only=True)
    #     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    #     callbacks=[logging, checkpoint, early_stopping]
    # else:
    #     callbacks=[logging]

    # model.save('my_model.h5')
    # print('model saved')
    for i in range(epochs):
        shuffle_idx = random.sample(range(6400), 6400)
        for j in range(6400 // batch_size):
            idx = shuffle_idx[j*batch_size:j*batch_size+batch_size]
            data_images, data_boxes = load_img_box(data_path, idx)
            image_data, boxes = process_data(data_images, data_boxes)
            print(image_data.shape)
            detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)
            verbose = (j % 50 == 0)
            model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)), verbose=verbose)
            if j % 100 == 0:
                model.save_weights(save_weights)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
