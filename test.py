import tensorflow as tf
import numpy as np
import cv2

import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append('./')
import data_loader
from settings import Settings
CONFIG = Settings(required_args=["gpu","config","checkpoint"])
scan_loader = data_loader.file_loader(CONFIG, test = True)

# Computes softmax
def softmax(x):
	e_x = np.exp(x)
	return e_x / np.expand_dims(np.sum(e_x, axis=2), axis=2)

# Erase line in stdout
def erase_line():
	sys.stdout.write("\033[F")

# Compute scores for a single image
def compute_iou_per_class(pred, label, mask, n_class):

	pred  = np.argmax(pred[...,0:n_class], axis=2) * mask
	label = label * mask

	ious = np.zeros(n_class)
	tps  = np.zeros(n_class)
	fns  = np.zeros(n_class)
	fps  = np.zeros(n_class)

	for cls_id in range(n_class):
		tp = np.sum(pred[label == cls_id] == cls_id)
		fp = np.sum(label[pred == cls_id] != cls_id)
		fn = np.sum(pred[label == cls_id] != cls_id)

		ious[cls_id] = tp/(tp+fn+fp+0.00000001)
		tps[cls_id] = tp
		fps[cls_id] = fp
		fns[cls_id] = fn

	return ious, tps, fps, fns

# Create a colored image with depth or label colors
def label_to_img(label_sm, depth, mask):
	img = np.zeros((label_sm.shape[0], label_sm.shape[1], 3))

	colors = np.array([[0,0,0],[78,205,196],[199,244,100],[255,107,107]])

	label = np.argmax(label_sm, axis=2)
	label = np.where(mask == 1, label, 0)

	for y in range(0,label.shape[0]):
		for x in range(0,label.shape[1]):
			if label[y,x] == 0:
				img[y,x,:] = [depth[y,x] * 255.0, depth[y,x] * 255.0, depth[y,x] * 255.0]
			else:
				img[y,x,:] = colors[label[y,x],:]

	return img / 255.0

# Export pointcloud with colored labels
def label_to_xyz(label_sm, data, mask, file):
	colors = np.array([[100,100,100],[78,205,196],[199,244,100],[255,107,107]])

	ys, xs = np.where(mask == 1)
	label = np.argmax(label_sm, axis=2)

	file = open(file, "w")
	for p in range(0, ys.shape[0]):
		x = xs[p]
		y = ys[p]
		l = label[y, x]
		file.write("{} {} {} {} {} {}\n".format(data[y, x, 0], data[y, x, 1], data[y, x, 2], colors[l, 0], colors[l, 1], colors[l, 2]))

	file.close()


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

# Read a single file
def read_example():
	points, neighbors, label = scan_loader.read_batch(training = False)
	groundtruth = np.argmax(label[:,:,:,0:CONFIG.N_CLASSES], axis=3)
	groundtruth = np.squeeze(groundtruth)
	mask = label[:,:,:,CONFIG.N_CLASSES+1] == 1
	mask = np.squeeze(mask)
	return points, neighbors, groundtruth, label[:,:,0:CONFIG.N_CLASSES], mask



# Run test routine
def test(checkpoint = None, display=False):

	# Which checkpoint should be tested
	if checkpoint is not None:
		CONFIG.TEST_CHECKPOINT = checkpoint

	# Create output dir if needed
	if not os.path.exists(CONFIG.TEST_OUTPUT_PATH):
		os.makedirs(CONFIG.TEST_OUTPUT_PATH)

	print("Processing dataset file \"{}\" for checkpoint {}:".format(CONFIG.TFRECORD_TEST, str(CONFIG.TEST_CHECKPOINT)))

	graph = tf.Graph()
	with tf.Session(graph=graph) as sess:
		loader = tf.train.import_meta_graph(CONFIG.OUTPUT_MODEL + "-" + str(CONFIG.TEST_CHECKPOINT) + ".meta")
		loader.restore(sess, CONFIG.OUTPUT_MODEL + "-" + str(CONFIG.TEST_CHECKPOINT))

		points     = graph.get_tensor_by_name("points_placeholder:0")
		neighbors  = graph.get_tensor_by_name("neighbors_placeholder:0")
		train_flag = graph.get_tensor_by_name("flag_placeholder:0")
		y          = graph.get_tensor_by_name("net/y:0")


		# Running network on each example
		line_num   = 1

		tps_sum = 0
		fns_sum = 0
		fps_sum = 0

		CONFIG.BATCH_SIZE = 1
		fill_length = len(str(scan_loader.test_len))
		for line_num in range(scan_loader.test_len):

			points_data, neighbors_data, groundtruth, label, mask = read_example()

			ref = np.reshape(points_data, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_DEPTH))
			img = ref
			groundtruth = data_loader.apply_mask(groundtruth, mask)

			# Inference
			data = sess.run(y, feed_dict = {points: points_data, neighbors: neighbors_data, train_flag: False})
			pred = softmax(data[0,:,:,:])

			bgr_image = cm.viridis(ref[:,:,3])[:,:,:3]*255
			viridis_image = np.zeros([CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH,3])
			viridis_image[:,:,0], viridis_image[:,:,1], viridis_image[:,:,2]= bgr_image[:,:,2], bgr_image[:,:,1], bgr_image[:,:,0]

			labels = np.argmax(pred, axis=2) * mask * 255

			dynamic_pixels = np.argwhere(labels==255)
			for pixel in dynamic_pixels:
				viridis_image[pixel[0], pixel[1], :] = [0, 0, 255]

			new_image = viridis_image.astype(np.uint8)

			cv2.imwrite(CONFIG.TEST_OUTPUT_PATH + str(line_num).zfill(fill_length) + '.png', new_image)

			iou, tps, fps, fns = compute_iou_per_class(pred, groundtruth, mask, CONFIG.N_CLASSES)

			tps_sum += tps
			fns_sum += fns
			fps_sum += fps

			print(" >> Processed file {}: IoUs {}".format(line_num, iou))

			line_num += 1


		ious = tps_sum.astype(np.float)/(tps_sum + fns_sum + fps_sum + 0.000000001)
		pr   = tps_sum.astype(np.float)/(tps_sum + fps_sum + 0.000000001)
		re   = tps_sum.astype(np.float)/(tps_sum + fns_sum + 0.000000001)

		output = "[{}] Accuracy:\n".format(checkpoint)
		for i in range(1, CONFIG.N_CLASSES):
			output += "\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}\n".format(pr[i], re[i], ious[i])
		output += "\n"

		return output


def ckpt_exists(ckpt):
	return os.path.isfile(CONFIG.OUTPUT_MODEL + "-" + str(ckpt) + ".meta")

if __name__ == "__main__":
	file = open("results_" + os.path.basename(CONFIG.CONFIG_NAME)[:-4] + ".txt", "w")

	ckpt = 40000
	# ckpt = CONFIG.SAVE_INTERVAL
	# while ckpt <= CONFIG.NUM_ITERS:
	output = test(checkpoint = ckpt)

	print(output)
	file.write(output)
	file.flush()

	# while not ckpt_exists(ckpt + CONFIG.SAVE_INTERVAL) and ckpt < CONFIG.NUM_ITERS:
	# 	print("Waiting for the next checkpoint ...")
	# 	time.sleep(60)

	# ckpt += CONFIG.SAVE_INTERVAL

	file.close()
