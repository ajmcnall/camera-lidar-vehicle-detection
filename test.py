"""
This is a script that can be used to test the model for your own dataset.
"""
import argparse
import random
import os
from glob import glob
import imghdr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
									 yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# ArgsParser
parser = argparse.ArgumentParser(description="test model some data.")
parser.add_argument(
	'--data_path', dest='data_path',
	help='path to test data',
	default='./Car_detection_data/rob599_dataset_deploy/test/',
	type=str)
parser.add_argument(
	'--class_num', dest='class_num',
	help='path to classes file',
	default=23,
	type=int)
parser.add_argument(
	'--weights_path', dest='weights_path',
	help='path to weights file',
	default='./trained_stage_2.h5',
	type=str)
parser.add_argument(
	'--save_path', dest='save_path',
	help='path to save file',
	default='./Car_detection_data/res.txt',
	type=str)
parser.add_argument(
	'--score_threshold', dest='score_threshold',
	help='threshold for score',
	default=0.3,
	type=float)

YOLO_ANCHORS = np.array(
	((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
	 (7.88282, 3.52778), (9.77052, 9.16828)))

def test_(args):
	df_pred = pd.DataFrame(columns=['guid/image', 'N'])
	baselist=[]
	countlist = []
	Ignore_class = [0, 9, 11, 14, 15, 16, 17, 21, 22]
	class_num = args.class_num

	anchors = YOLO_ANCHORS
	model_body = create_model(anchors, class_num)
	model_body.load_weights(args.weights_path)

	yolo_outputs = yolo_head(model_body.output, anchors, class_num)
	input_image_shape = K.placeholder(shape=(2, ))

	boxes, scores, classes = yolo_eval(
		yolo_outputs, input_image_shape, score_threshold=args.score_threshold, iou_threshold=0.4)
	sess = K.get_session()
	pic_num = 0
	for path in os.listdir(args.data_path):
		test_path = os.path.join(args.data_path, path)
		for image_file in os.listdir(test_path):
			try:
				image_type = imghdr.what(os.path.join(test_path, image_file))
				if not image_type:
					continue
			except IsADirectoryError:
				continue
			pic_num += 1
			image = PIL.Image.open(os.path.join(test_path, image_file))
			resized_image = image.resize((416, 416), PIL.Image.BICUBIC)
			image_data = np.array(resized_image, dtype='float32')
			image_data /= 255.
			image_data = np.expand_dims(image_data, 0)
			out_boxes, out_scores, out_classes = sess.run(
					[boxes, scores, classes],
					feed_dict={
						model_body.input: image_data,
						input_image_shape: [image.size[1], image.size[0]],
						K.learning_phase(): 0
					})
			baselist.append(path + '/' + image_file[:4])
			count = 0
			print(len(out_classes))
			for out_class in out_classes:
					count += out_class
			countlist.append(count)
			print(baselist[-1], countlist[-1], pic_num)
	df_pred['guid/image'] = baselist
	df_pred['N'] = countlist
	df_pred.to_csv(args.save_path, index=False)
	sess.close()


def create_model(anchors, class_num):

	# Create model input layers.
	image_input = Input(shape=(416, 416, 3))

	yolo_model = yolo_body(image_input, len(anchors), class_num)
	topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
	final_layer = Conv2D(len(anchors)*(5+class_num), (1, 1), activation='linear')(topless_yolo.output)
	model_body = Model(image_input, final_layer)

	return model_body

if __name__ == '__main__':
	args = parser.parse_args()
	for idx, threshold in enumerate([2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]):
		args.score_threshold = threshold
		args.save_path = './Car_detection_data/res' + str(idx) +'.txt'
		test_(args)
