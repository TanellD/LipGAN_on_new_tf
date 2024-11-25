from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2D, Conv3D, BatchNormalization, Activation, \
						Concatenate, AvgPool2D, Input, MaxPool2D, UpSampling2D, Add, \
						ZeroPadding2D, ZeroPadding3D, Lambda, Reshape, Flatten, LeakyReLU, GroupNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K
from keras import ops
from tensorflow.keras.losses import MSE

def conv_block(x, num_filters, kernel_size=3, strides=2, padding='same'):
	x = Conv2D(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = GroupNormalization(groups=-1)(x)
	x = LeakyReLU(negative_slope=.2)(x)
	return x

class L2Layer(tf.keras.layers.Layer):
	def __init__(self):
		super(L2Layer, self).__init__()
		self.support_masking=True
	def call(self, inputs, mask=None):
		return tf.math.l2_normalize(inputs, axis=1) 
def create_model(args, mel_step_size):
	############# encoder for face/identity
	
	input_face = Input(shape=(args.img_size, args.img_size, 3), name="input_face_disc")
	x = conv_block(input_face, 64, 7)
	x = conv_block(x, 128, 5)
	x = conv_block(x, 256, 3)
	x = conv_block(x, 512, 3)
	x = conv_block(x, 512, 3)
	x = Conv2D(filters=512, kernel_size=3, strides=1, padding="valid")(x)
	face_embedding = Flatten() (x)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x = conv_block(input_audio, 32, strides=1)
	x = conv_block(x, 64, strides=3)	#27X9
	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = conv_block(x, 256, strides=3)	#3X3
	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, strides=1)

	audio_embedding = Flatten() (x)
	# L2-normalize before taking L2 distance
	# l2_normalize = Lambda(lambda x: K.l2_normalize(x, axis=1)) 

	face_embedding = L2Layer() (face_embedding)
	audio_embedding = L2Layer() (audio_embedding)
	d = Lambda(lambda x: tf.math.sqrt(tf.math.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)), output_shape=(None, 512)) ([face_embedding,
																		audio_embedding])
	model = Model(inputs=[input_face, input_audio], outputs=[d])

	# model.summary()
	model.compile(optimizer=Adam(learning_rate=args.lr), loss=contrastive_loss) 
	
	return model
	

if __name__ == '__main__':
	model = create_model()
