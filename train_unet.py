from os import listdir, path
import numpy as np
import scipy
import cv2
import os, sys
from generator import create_model_residual, create_model
from keras.callbacks import ModelCheckpoint, Callback
from glob import glob
import pickle, argparse
import tensorflow as tf

half_window_size = 4
mel_step_size = 27

def frame_id(fname):
	return int(os.path.basename(fname).split('.')[0])

def choose_ip_frame(frames, gt_frame):
	selected_frames = [f for f in frames if np.abs(frame_id(gt_frame) - frame_id(f)) >= 6]
	return np.random.choice(selected_frames)

def get_audio_segment(center_frame, spec):
	center_frame_id = frame_id(center_frame)
	start_frame_id = center_frame_id - half_window_size

	start_idx = int((80./25.) * start_frame_id) # 25 is fps of LRS2
	end_idx = start_idx + mel_step_size

	return spec[:, start_idx : end_idx] if end_idx <= spec.shape[1] else None

class DataGenerator(object):
	def __init__(self, all_images, batch_size, img_size, mel_step_size):

		self.img_gt_batch = []
		self.img_ip_batch = []
		self.mel_batch = []
		self.all_images = all_images
		self.batch_size = batch_size
		self.img_size = img_size
		self.mel_step_size = mel_step_size
		self.i = -1

	def next(self):
		
		self.img_ip_batch.clear()
		self.mel_batch.clear()
		self.img_gt_batch.clear()
		img_gt_batch_arr, img_ip_batch_arr, mel_batch_arr, img_gt_batch_masked_arr = None, None, None, None
		del img_gt_batch_arr, img_ip_batch_arr, mel_batch_arr, img_gt_batch_masked_arr
		# for img_name in all_images:
		while len(self.img_gt_batch) < self.batch_size:	
			self.i += 1
		# For each image in the batch
			img_name = self.all_images[self.i]
			gt_fname = os.path.basename(img_name)
			dir_name = img_name.replace(gt_fname, '')
			frames = glob(dir_name + '/*.jpg')

			if len(frames) < 12:
				continue

			mel_fname = dir_name + '/mels.npz'
			try:
				mel = np.load(mel_fname)['spec']
			except:
				continue

			mel = get_audio_segment(gt_fname, mel)

			if mel is None or mel.shape[1] != self.mel_step_size:
				continue

			if np.isnan(mel.flatten()).sum() > 0:
				continue	

			img_gt = cv2.imread(img_name)
			img_gt = cv2.resize(img_gt, (self.img_size, self.img_size))

			ip_fname = choose_ip_frame(frames, gt_fname)
			img_ip = cv2.imread(ip_fname)
			img_ip = cv2.resize(img_ip, (self.img_size, self.img_size))

			self.img_gt_batch.append(img_gt)
			self.img_ip_batch.append(img_ip)
			self.mel_batch.append(mel)

				
		img_gt_batch_arr = np.asarray(self.img_gt_batch)
		img_ip_batch_arr = np.asarray(self.img_ip_batch)
		mel_batch_arr = np.expand_dims(np.asarray(self.mel_batch), 3)
		img_gt_batch_masked_arr = img_gt_batch_arr.copy()
		img_gt_batch_masked_arr[:, self.img_size//2:,...] = 0.
		img_ip_batch_arr = np.concatenate([img_ip_batch_arr, img_gt_batch_masked_arr], axis=3)
		yield (img_ip_batch_arr / 255.0, mel_batch_arr), img_gt_batch_arr / 255.0
		
	
	def __next__(self):
		return self.next()
	def __iter__(self):
		return self

parser = argparse.ArgumentParser(description='Keras implementation of LipGAN')

parser.add_argument('--data_root', type=str, help='LRS2 preprocessed dataset root to train on', required=True)
parser.add_argument('--logdir', type=str, help='Folder to store checkpoints', default='logs/')

parser.add_argument('--model', type=str, help='Model name to use: basic|residual', default='residual')
parser.add_argument('--resume', help='Path to weight file to load into the model', default=None)
parser.add_argument('--checkpoint_name', type=str, help='Checkpoint filename to use while saving', 
						default='unet.weights.h5')
parser.add_argument('--checkpoint_freq', type=int, help='Frequency of checkpointing', default=1)

parser.add_argument('--n_gpu', type=int, help='Number of GPUs to use', default=1)
parser.add_argument('--batch_size', type=int, help='Single GPU batch size', default=32)
parser.add_argument('--lr', type=float, help='Initial learning rate', default=1e-3)
parser.add_argument('--img_size', type=int, help='Size of input image', default=96)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=2)
parser.add_argument('--mel_step_size', type=int, help='Number of mel', default=27)
parser.add_argument('--all_images', default='filenames.pkl', help='Filename for caching image paths')
args = parser.parse_args()

if path.exists(path.join(args.logdir, args.all_images)):
	args.all_images = pickle.load(open(path.join(args.logdir, args.all_images), 'rb'))
else:
	all_images = glob(path.join("{}/*/*.jpg".format(args.data_root)))
	pickle.dump(all_images, open(path.join(args.logdir, args.all_images), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	args.all_images = all_images
	
print ("Will be training on {} images".format(len(args.all_images)))

if args.model == 'residual':
	model = create_model_residual(args, mel_step_size)
else:
	model = create_model(args, mel_step_size)

if args.resume:
	model.load_weights(args.resume)
	print('Resuming from : {}'.format(args.resume))

args.batch_size = args.n_gpu * args.batch_size
# train_datagen = datagen(args)
class WeightsSaver(Callback):
	def __init__(self, N, weight_path):
		self.N = N
		self.batch = 0
		self.weight_path = weight_path

	def on_batch_end(self, batch, logs={}):
		self.batch += 1
		if self.batch % self.N == 0:
			self.model.save_weights(self.weight_path)

callbacks_list = [WeightsSaver(args.checkpoint_freq, path.join(args.logdir, args.checkpoint_name))]

def generator_fn(all_images, batch_size, img_size, mel_step_size):
    data_gen = DataGenerator(all_images, batch_size, img_size, mel_step_size)
    for data in data_gen:
        yield list(data)[0]

# Create the tf.data.Dataset
def create_tf_dataset(all_images, batch_size, img_size, mel_step_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator_fn(all_images, batch_size, img_size, mel_step_size),
        output_signature=(
            (tf.TensorSpec(shape=[batch_size, img_size, img_size, 6], dtype=tf.float32),  # img_ip_batch + img_gt_batch_masked
             tf.TensorSpec(shape=[batch_size, 80, mel_step_size, 1], dtype=tf.float32)),  # mel_batch
            tf.TensorSpec(shape=[batch_size, img_size, img_size, 3], dtype=tf.float32)  # img_gt_batch
        )
    )
    return dataset

with open('logs/filenames.pkl', 'rb') as f:
    all_images = pickle.load(f)

# Parameters

batch_size = 32
img_size = 96
mel_step_size = 27
dataset = create_tf_dataset(all_images, batch_size, img_size, mel_step_size)

model.fit(dataset.take(100), epochs=2, steps_per_epoch=100)


model.save_weights(f"logs/trained_unet.weights.h5")

