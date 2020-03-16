import sys
import numpy as np
import cv2
import os
import math
import time
from scipy import interpolate
import random
import shutil
from settings import Settings
import tensorflow as tf


# Takes a sequence of channels and returns the corresponding indices in the rangeimage
def seq_to_idx(seq):
	idx = []
	if "x" in seq:
		idx.append(0)
	if "y" in seq:
		idx.append(1)
	if "z" in seq:
		idx.append(2)
	if "r" in seq:
		idx.append(3)
	if "d" in seq:
		idx.append(4)

	return np.array(idx, dtype=np.intp)



def lindepth_to_mask(depth_linear, img_height, img_width):
	return np.reshape(depth_linear, (img_height, img_width, 1)) > 0


def clip_normalize(data, interval, log_transformed=False):
	data_clip = np.clip(data, interval[0], interval[1])
	if log_transformed:
		#return (np.log(data_clip) - np.log(interval[0])) / (np.log(interval[1]) - np.log(interval[0]))
		return data_clip
	else:
		return (data_clip - interval[0]) / (interval[1] - interval[0])


def clip_mask_normalize(data, mask, interval, log_transformed=False):
	outerval = np.logical_and(data < interval[1], data > interval[0])
	mask = np.logical_and(mask, outerval)
	data_clip = np.clip(data, interval[0], interval[1])

	if log_transformed:
		#return (np.log(data_clip) - np.log(interval[0])) / (np.log(interval[1]) - np.log(interval[0])), mask
		return np.log(data_clip), mask
	else:
		return (data_clip - interval[0]) / (interval[1] - interval[0]), mask


def fill_sky(data, mask, new_val):
	ret, labels = cv2.connectedComponents(np.asarray(mask == 0).astype(np.uint8))
	sky_label = labels[0, math.floor(mask.shape[1] / 2)]

	cv2.imwrite("./validation/test.png", labels)
	for c in range(data.shape[2]):
		data[:,:,c] = np.where(labels == sky_label, new_val, data[:,:,c])

	return data



def apply_mask(data, mask):
	tmp = np.zeros((data.shape[0], data.shape[1]))

	if len(data.shape) == 2:
		data[np.squeeze(mask)] == 0
	else:
		for c in range(data.shape[2]):
			data[:,:,c] = np.where(np.squeeze(mask) == 1, data[:,:,c], tmp)

	return data


def ri_to_depth_height_mask(ri, depth_clip, height_clip):
	mask = ri[:,:,0] > 0

	depth, mask = clip_mask_normalize(np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2), mask, depth_clip, log_transformed = True)

	height, mask = clip_mask_normalize(ri[:,:,2], mask, height_clip)

	img = apply_mask(np.dstack((depth, height)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_depth_height_intensity_mask(ri, depth_clip, height_clip):
	mask = ri[:,:,0] > 0

	depth, mask = clip_mask_normalize(np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2), mask, depth_clip, log_transformed = True)

	height, mask = clip_mask_normalize(ri[:,:,2], mask, height_clip)

	ref = ri[:,:,3]

	img = apply_mask(np.dstack((depth, height, ref)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_depth_height_intensity_mask_noclip(ri, depth_clip, height_clip):
	mask = ri[:,:,0] > 0

	depth = np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2)

	height = ri[:,:,2]

	ref = ri[:,:,3]

	img = apply_mask(np.dstack((depth, height, ref)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_depth_height_mask_noclip(ri):
	mask = ri[:,:,0] > 0

	depth = np.sqrt(ri[:,:,0]**2 + ri[:,:,1]**2)

	height = ri[:,:,2]

	img = apply_mask(np.dstack((depth, height)).astype(np.float32), mask)

	mask = mask

	return img, mask

def ri_to_xyz_mask(ri):
	mask = ri[:,:,0] > 0

	img = ri[:,:,0:3]

	mask = mask

	return img, mask

def ri_to_xyz_intensity_depth_mask(ri):
	mask = ri[:,:,0] > 0

	img = ri[:,:,0:5]

	mask = mask

	return img, mask


def interp_data(d, mask):
	interp_output = np.zeros(d.shape)
	x = np.arange(0, d.shape[1])
	y = np.arange(0, d.shape[0])

	xx, yy = np.meshgrid(x, y)

	x1 = xx[mask]
	y1 = yy[mask]
	for c in range(d.shape[2]):
		newarr = d[:,:,c]
		newarr = newarr[mask]
		interp_output[:,:,c] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')

	return interp_output

def pointnetize_slow(groundtruth, n_size=[3, 3]):
	y_offset    = int(math.floor(n_size[0] / 2))
	x_offset    = int(math.floor(n_size[1] / 2))
	n_len       = n_size[0] * n_size[1]
	mid_offset  = int(math.floor(n_len / 2))
	n_indices   = np.delete(np.arange(n_len), mid_offset)

	groundtruth_pad = np.pad(groundtruth, ((y_offset, y_offset),(x_offset, x_offset), (0, 0)), "symmetric")

	n_output = np.zeros((groundtruth.shape[0], groundtruth.shape[1], n_len - 1, groundtruth.shape[2]))
	p_output = np.zeros((groundtruth.shape[0], groundtruth.shape[1], 1, groundtruth.shape[2]))

	valid = 0
	mask = 0
	for y in range(0, groundtruth.shape[0]):
		for x in range(0, groundtruth.shape[1]):
			patch = groundtruth_pad[y:y+n_size[0], x:x+n_size[1],:]
			lin_patch = np.reshape(patch, (n_len, -1))

			if lin_patch[mid_offset,0] != 0: # If center pixel is not empty
				valid = valid + 1
				p = lin_patch[mid_offset, :]
				n = lin_patch[n_indices, :]

				mask_filled = n[:,0] != 0
				mask = mask + np.sum(mask_filled.flatten())
				mask_not_filled = n[:,0] == 0

				n[mask_filled, 0:3] = n[mask_filled, 0:3] - p[0:3] # Defined points in local coordinates
				n[mask_not_filled,:] = 0
				n_output[y,x,:,:] = n
				p_output[y,x,:,:] = p
	return p_output, n_output

def gt_to_label(groundtruth, mask, n_classes):

	# Compute weigthed mask
	contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

	if np.amax(groundtruth) > n_classes-1:
		print("[WARNING] There are more classes than expected !")

	for c in range(1, int(np.amax(groundtruth))+1):
		channel = (groundtruth == c).astype(np.float32)
		gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
		gt_dilate = gt_dilate - channel
		contours = np.logical_or(contours, gt_dilate == 1.0)

	contours = contours.astype(np.float32) * mask

	dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

	weight_map = 0.1 + 1.0 * np.exp(- dist / (2.0 * 3.0**2.0))
	weight_map = weight_map * mask

	# Create output label for training
	label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], n_classes + 1))
	for y in range(groundtruth.shape[0]):
		for x in range(groundtruth.shape[1]):
			label[y, x, int(groundtruth[y, x])] = 1.0

	label[:,:,n_classes] = weight_map

	return label

class file_loader():
	"""Image database."""
	def __init__(self, settings):
		self._image_set = []
		self._image_idx = []
		self._data_root_path = []
		self._settings = settings
		self._train_image_idx, self._val_image_idx = self._load_image_set_idx()

		## batch reader ##
		self._perm_train_idx = None
		self._perm_val_idx = None
		self._cur_train_idx = 0
		self._cur_val_idx = 0
		self._shuffle_train_image_idx()
		self._shuffle_val_image_idx()
		self._n_size = settings.N_SIZE
		self._y_offset    = int(math.floor(self._n_size[0] / 2))
		self._x_offset    = int(math.floor(self._n_size[1] / 2))
		self._n_len       = self._n_size[0] * self._n_size[1]
		self._mid_offset  = int(math.floor(self._n_len / 2))
		self._n_indices   = np.delete(np.arange(self._n_len), self._mid_offset)

	def _load_image_set_idx(self):
		train_folders = open(self._settings.DATA_ROOT_PATH + "train.txt", "r")
		val_folders = open(self._settings.DATA_ROOT_PATH + "val.txt", "r")

		train_image_idx = []
		val_image_idx = []

		for subfolder in train_folders:
			npy_files_in_subfolder = []
			subfolder = subfolder.rstrip("\n")
			for f in os.listdir(subfolder):
				npy_file = os.path.join(subfolder, f)
				assert os.path.exists(npy_file), \
					'File does not exist: {}'.format(npy_file)
				if (os.path.isfile(npy_file) and npy_file[-3:] == "npy"):
					train_image_idx.append(npy_file)

		for subfolder in val_folders:
			subfolder = subfolder.rstrip("\n")
			npy_files_in_subfolder = []
			for f in os.listdir(subfolder):
				npy_file = os.path.join(subfolder, f)
				assert os.path.exists(npy_file), \
					'File does not exist: {}'.format(npy_file)
				if (os.path.isfile(npy_file) and npy_file[-3:] == "npy"):
					val_image_idx.append(npy_file)
		return train_image_idx, val_image_idx

	@property
	def image_idx(self):
		return self._image_idx

	@property
	def image_set(self):
		return self._image_set

	@property
	def data_root_path(self):
		return self._data_root_path

	def _pointnetize(self, groundtruth):

		groundtruth_pad = np.pad(groundtruth, ((self._y_offset, self._y_offset),(self._x_offset, self._x_offset), (0, 0)), "symmetric")

		neighbors = np.zeros((self._n_len,groundtruth.shape[0], groundtruth.shape[1],groundtruth.shape[2]), dtype = np.float32)
		i = 0
		for y_shift in [1,0,-1]:
			for x_shift in [1,0,-1]:
				neighbors[i,:,:,:] = np.roll(groundtruth_pad, [y_shift, x_shift],axis=(0,1))[self._y_offset:-self._y_offset,self._x_offset:-self._x_offset,:]
				i = i + 1

		point_subt = np.expand_dims(groundtruth, axis = 0)
		point_subt = np.repeat(point_subt, self._n_len, axis = 0)
		point_subt[:,:,:,3:] = 0
		point_subt[self._mid_offset,:,:,:] = 0
		x_tensor = neighbors[:,:,:,0] != 0
		x_tensor = np.expand_dims(x_tensor, axis = -1)
		x_tensor = np.repeat(x_tensor, groundtruth.shape[2], axis=-1)
		x_bin = x_tensor[self._mid_offset,:,:,0] != 0
		x_bin = np.expand_dims(x_bin, axis=-1)
		x_bin = np.repeat(x_bin,groundtruth.shape[2], axis=-1)
		x_bin = np.expand_dims(x_bin, axis=0)
		x_bin = np.repeat(x_bin,self._n_len, axis=0)
		keep = np.logical_and(x_bin, x_tensor)
		n_diff = neighbors - point_subt
		end = np.multiply(keep, n_diff)
		end = np.transpose(end, (1, 2, 0, 3))
		n_output = end[:,:,self._n_indices,:]
		p_output = end[:,:,self._mid_offset,:]
		p_output = np.expand_dims(p_output, axis=2)
		# n_output = np.zeros((groundtruth.shape[0], groundtruth.shape[1], n_len - 1, groundtruth.shape[2]))
		# p_output = np.zeros((groundtruth.shape[0], groundtruth.shape[1], 1, groundtruth.shape[2]))
		return p_output, n_output

	def _shuffle_train_image_idx(self):
		self._perm_train_idx = [self._train_image_idx[i] for i in
		    np.random.permutation(np.arange(len(self._train_image_idx)))]
		self._cur_train_idx = 0

	def _shuffle_val_image_idx(self):
		self._perm_val_idx = [self._val_image_idx[i] for i in
		    np.random.permutation(np.arange(len(self._val_image_idx)))]
		self._cur_val_idx = 0

	def read_batch(self, training, shuffle=True):
		"""Read a batch of lidar data including labels. Data formated as numpy array
		of shape: height x width x {x, y, z, intensity, range, label}.
		Args:
		  shuffle: whether or not to shuffle the dataset
		Returns:
		  lidar_per_batch: LiDAR input. Shape: batch x height x width x 5.
		  lidar_mask_per_batch: LiDAR mask, 0 for missing data and 1 otherwise.
		    Shape: batch x height x width x 1.
		  label_per_batch: point-wise labels. Shape: batch x height x width.
		  weight_per_batch: loss weights for different classes. Shape:
		    batch x height x width
		"""
		settings = self._settings
		batch_idx = []
		if(training):
		    if shuffle:
		      if self._cur_train_idx + settings.BATCH_SIZE >= len(self._train_image_idx):
		        self._shuffle_train_image_idx()
		      batch_idx = self._perm_train_idx[self._cur_train_idx:self._cur_train_idx+settings.BATCH_SIZE]
		      self._cur_train_idx += settings.BATCH_SIZE
		    else:
		      if self._cur_train_idx + settings.BATCH_SIZE >= len(self._train_image_idx):
		        batch_idx = self._train_image_idx[self._cur_train_idx:] \
		            + self._train_image_idx[:self._cur_train_idx + settings.BATCH_SIZE-len(self._train_image_idx)]
		        self._cur_train_idx += settings.BATCH_SIZE - len(self._train_image_idx)
		      else:
		        batch_idx = self._train_image_idx[self._cur_train_idx:self._cur_train_idx+settings.BATCH_SIZE]
		        self._cur_train_idx += settings.BATCH_SIZE

		if( not training):
		    if shuffle:
		      if self._cur_val_idx + settings.BATCH_SIZE >= len(self._val_image_idx):
		        self._shuffle_val_image_idx()
		      batch_idx = self._perm_val_idx[self._cur_val_idx:self._cur_val_idx+settings.BATCH_SIZE]
		      self._cur_val_idx += settings.BATCH_SIZE
		    else:
		      if self._cur_val_idx + settings.BATCH_SIZE >= len(self._val_image_idx):
		        batch_idx = self._val_image_idx[self._cur_val_idx:] \
		            + self._val_image_idx[:self._cur_val_idx + settings.BATCH_SIZE-len(self._val_image_idx)]
		        self._cur_val_idx += settings.BATCH_SIZE - len(self._val_image_idx)
		      else:
		        batch_idx = self._val_image_idx[self._cur_val_idx:self._cur_val_idx+settings.BATCH_SIZE]
		        self._cur_val_idx += settings.BATCH_SIZE

		points_raw = np.empty([0, settings.IMAGE_HEIGHT , settings.IMAGE_WIDTH, 1 , 5], np.float32)
		neighbors_raw = np.empty([0, settings.IMAGE_HEIGHT , settings.IMAGE_WIDTH,   settings.N_LEN, 5], np.float32)
		labels_raw = np.empty([0, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.N_CLASSES +   2], np.float32)

		image_idx = []

		for idx in batch_idx:
			# load data
			# loading from npy is 30x faster than loading from pickle
			t = time.time()
			data = np.load(idx).astype(np.float32, copy=False)

			# print("load: ", t-time.time())
			#data = np.ones([settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 6]) * 0.11
			if(settings.IMAGE_WIDTH == 512):
				data = data[:,0::4,:]
			if(settings.IMAGE_WIDTH == 1024):
				data = data[:,0::2,:]

			if settings.AUGMENTATION:
				if settings.RANDOM_FLIPPING:
					if np.random.rand() > 0.5:
						# flip y
						data = data[:, ::-1, :]
						data[:, :, 1] *= -1

			t = time.time()
			#p, n = pointnetize_slow(data[:,:,0:5], n_size=settings.N_SIZE)
			p, n = self._pointnetize(data[:,:,0:5])
			# print("pointnetize: ", t-time.time())
			t = time.time()
			mask = data[:,:,0] != 0

			groundtruth = apply_mask(data[:,:,5], mask)
			# print("gt shape:", groundtruth.shape)
			# print("max: ", groundtruth.max().max())

			# Compute weigthed mask
			contours = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

			if np.amax(groundtruth) > settings.N_CLASSES-1:
				print("[WARNING] There are more classes than expected !")

			for c in range(1, int(np.amax(groundtruth))+1):
				channel = (groundtruth == c).astype(np.float32)
				gt_dilate = cv2.dilate(channel, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
				gt_dilate = gt_dilate - channel
				contours = np.logical_or(contours, gt_dilate == 1.0)

			contours = contours.astype(np.float32) * mask

			dist = cv2.distanceTransform((1 - contours).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
			# print("dist: ", t-time.time())

			# Create output label for training
			label = np.zeros((groundtruth.shape[0], groundtruth.shape[1], settings.N_CLASSES + 2), dtype=np.float32)
			for y in range(groundtruth.shape[0]):
				for x in range(groundtruth.shape[1]):
					label[y, x, int(groundtruth[y, x])] = 1.0

			label[:,:,settings.N_CLASSES]   = dist
			label[:,:,settings.N_CLASSES+1] = mask
			p = np.expand_dims(p, axis = 0)
			n = np.expand_dims(n, axis = 0)
			label = np.expand_dims(label, axis = 0)
			points_raw = np.append(points_raw, p, axis=0)
			neighbors_raw = np.append(neighbors_raw, n, axis=0)
			labels_raw = np.append(labels_raw, label, axis=0)
			# print("rest: ", t-time.time())
		# points = tf.reshape(tf.convert_to_tensor(points_raw, dtype=tf.float32), [batch_size, settings.IMAGE_HEIGHT * settings.IMAGE_WIDTH, 1 , 5])
		points = np.reshape(points_raw, [settings.BATCH_SIZE, settings.IMAGE_HEIGHT * settings.IMAGE_WIDTH, 1 , 5])
		# neighbors = tf.reshape(tf.convert_to_tensor(neighbors_raw, dtype=tf.float32), [batch_size, settings.IMAGE_HEIGHT * settings.IMAGE_WIDTH,   settings.N_LEN, 5])
		neighbors = np.reshape(neighbors_raw, [settings.BATCH_SIZE, settings.IMAGE_HEIGHT * settings.IMAGE_WIDTH,   settings.N_LEN, 5])
		# labels = tf.reshape(tf.convert_to_tensor(labels_raw, dtype=tf.float32), [batch_size, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.N_CLASSES +   2])
		labels = np.reshape(labels_raw, [settings.BATCH_SIZE, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.N_CLASSES +   2])

		points = np.take(points, seq_to_idx(settings.CHANNELS), axis=3)
		neighbors = np.take(neighbors, seq_to_idx(settings.CHANNELS), axis=3)
		points = points.astype(np.float32)
		neighbors = neighbors.astype(np.float32)
		labels = labels.astype(np.float32)

		return points, neighbors, labels
