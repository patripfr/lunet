import sys
import tensorflow as tf
import numpy as np
import cv2
import csv
import os
import glob
import math
import matplotlib.pyplot as plt


# SQUEEZESEG DATASET
# Channels description:
# 0: X
# 1: Y
# 2: Z
# 3: REFLECTANCE
# 4: DEPTH
# 5: LABEL


# imports Settings manager
sys.path.append('./')
import data_loader
from settings import Settings
CONFIG = Settings(required_args=["config"])

# Generates tfrecords for training
def make_tfrecord():

	# Creates each tfrecord (train and val)

	list_subfolders_with_paths = [f[0] for f in os.walk(CONFIG.DATA_ROOT_PATH)]
	train_folder_name = open(CONFIG.DATA_ROOT_PATH + "train.txt", "r")
	val_folder_name = open(CONFIG.DATA_ROOT_PATH + "val.txt", "r")
	test_folder_name = open(CONFIG.DATA_ROOT_PATH + "test.txt", "r")
	train_image_idx = []
	val_image_idx = []
	test_image_idx = []
	for subfolder in train_folder_name:
		npy_files_in_subfolder = []
		subfolder = subfolder.rstrip("\n")
		for f in os.listdir(subfolder):
			npy_file = os.path.join(subfolder, f)
			assert os.path.exists(npy_file), \
				'File does not exist: {}'.format(npy_file)
			if (os.path.isfile(npy_file) and npy_file[-3:] == "npy"):
				train_image_idx.append(npy_file)
	for subfolder in val_folder_name:
		subfolder = subfolder.rstrip("\n")
		npy_files_in_subfolder = []
		for f in os.listdir(subfolder):
			npy_file = os.path.join(subfolder, f)
			assert os.path.exists(npy_file), \
				'File does not exist: {}'.format(npy_file)
			if (os.path.isfile(npy_file) and npy_file[-3:] == "npy"):
				val_image_idx.append(npy_file)
	for subfolder in test_folder_name:
		subfolder = subfolder.rstrip("\n")
		npy_files_in_subfolder = []
		for f in os.listdir(subfolder):
			npy_file = os.path.join(subfolder, f)
			assert os.path.exists(npy_file), \
				'File does not exist: {}'.format(npy_file)
			if (os.path.isfile(npy_file) and npy_file[-3:] == "npy"):
				test_image_idx.append(npy_file)

	for dataset in ["train", "val", "test"]:
	#for dataset in ["test"]:

		# Get path
		dataset_ouput = ""
		if (dataset == "train"):
			dataset_output = CONFIG.TFRECORD_TRAIN
		if (dataset == "val"):
			dataset_output = CONFIG.TFRECORD_VAL
		if (dataset == "test"):
			dataset_output = CONFIG.TFRECORD_TEST

		with tf.python_io.TFRecordWriter(dataset_output) as writer:

			file_list_name = open(dataset_output + ".txt", "w")

			if dataset == "train":
				file_list = train_image_idx
			if dataset == "val":
				file_list = val_image_idx
			if dataset == "test":
				file_list = test_image_idx

			# Going through each example
			line_num = 1
			for file in file_list:

				augmentation_list = ["normal"] if dataset is "val" else CONFIG.AUGMENTATION

				# Augmentation settings
				for aug_type in augmentation_list:

					print("[{}] >> Processing file \"{}\" ({}), with augmentation : {}".format(dataset, file[:-1], line_num, aug_type))

					# Load labels
					data = np.load(file)

					data = data[:,0::4,:]

					mask = data[:,:,0] != 0

					#data = data_loader.interp_data(data[:,:,0:5], mask)

					p, n = data_loader.pointnetize(data[:,:,0:5], n_size=CONFIG.N_SIZE)

					groundtruth = data_loader.apply_mask(data[:,:,5], mask)

					# Compute weigthed mask
					contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

					if np.amax(groundtruth) > CONFIG.N_CLASSES-1:
						print("[WARNING] There are more classes than expected !")

					for c in range(1, int(np.amax(groundtruth))+1):
						channel = (groundtruth == c).astype(np.float32)
						gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
						gt_dilate = gt_dilate - channel
						contours = np.logical_or(contours, gt_dilate == 1.0)

					contours = contours.astype(np.float32) * mask

					dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

					# Create output label for training
					label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], CONFIG.N_CLASSES + 2))
					for y in range(groundtruth.shape[0]):
						for x in range(groundtruth.shape[1]):
							label[y, x, int(groundtruth[y, x])] = 1.0

					label[:,:,CONFIG.N_CLASSES]   = dist
					label[:,:,CONFIG.N_CLASSES+1] = mask

					# Serialize example
					n_raw = n.astype(np.float32).tostring()
					p_raw = p.astype(np.float32).tostring()
					label_raw = label.astype(np.float32).tostring()

					# Create tf.Example
					example = tf.train.Example(features=tf.train.Features(feature={
							'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
							'neighbors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[n_raw])),
							'points': tf.train.Feature(bytes_list=tf.train.BytesList(value=[p_raw]))}))

					# Adding Example to tfrecord
					writer.write(example.SerializeToString())

					file_list_name.write(file +"\n")

				line_num += 1

			print("Process finished, stored {} entries in \"{}\"".format(line_num-1, dataset_output))

			file_list_name.close()

	print("All files created.")

if __name__ == "__main__":
	make_tfrecord()
