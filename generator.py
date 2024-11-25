from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2DTranspose, Conv2D, BatchNormalization, \
						Activation, Concatenate, Input, MaxPool2D,\
						UpSampling2D, ZeroPadding2D, Lambda, Add

from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from keras.utils import plot_model
import tensorflow as tf
# from discriminator import contrastive_loss

def contrastive_loss(y_true, y_pred):
    margin = 1.
    loss = (1. - y_true) * tf.square(y_pred) + y_true * tf.square(tf.maximum(0., margin - y_pred))
    return tf.reduce_mean(loss)

class GAN(keras.Model):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, g_loss_fn, d_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def train_step(self, data):
        (dummy_faces,audios), real_images = data
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        # random_latent_vectors = keras.random.normal(
        #     shape=(batch_size, self.latent_dim, self.latent_dim, ), seed=self.seed_generator
        # )

        # Decode them to fake images
        generated_images = self.generator((dummy_faces, audios))

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_audios = tf.concat([audios, audios], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * keras.random.uniform(
            tf.shape(labels), seed=self.seed_generator
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator((combined_images, combined_audios))
            d_loss = self.d_loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # Sample random points in the latent space
        # random_latent_vectors = keras.random.normal(
        #     shape=(batch_size, self.latent_dim, self.latent_dim, 3), seed=self.seed_generator
        # )

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated = self.generator((dummy_faces, audios))
            predictions = self.discriminator((generated, audios))
            g_adv_loss = self.d_loss_fn(misleading_labels, predictions)
            g_c_loss = self.g_loss_fn(real_images, generated)
            g_loss = 0.1*g_adv_loss + g_c_loss
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply(grads, self.generator.trainable_weights)


        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }
	


def conv_block(x, num_filters, kernel_size=3, strides=1, padding='same', act=True):
	x = Conv2D(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)
	if act:
		x = Activation('relu')(x)
	return x

def conv_t_block(x, num_filters, kernel_size=3, strides=2, padding='same'):
	x = Conv2DTranspose(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)
	x = Activation('relu')(x)

	return x

def create_model(args, mel_step_size):
	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face")

	identity_mapping = conv_block(input_face, 32, kernel_size=11) # 96x96
	x1_face = conv_block(identity_mapping, 64, kernel_size=7, strides=2) # 48x48
	x2_face = conv_block(x1_face, 128, 5, 2) # 24x24
	x3_face = conv_block(x2_face, 256, 3, 2) #12x12
	x4_face = conv_block(x3_face, 512, 3, 2) #6x6
	x5_face = conv_block(x4_face, 512, 3, 2) #3x3
	x6_face = conv_block(x5_face, 512, 3, 1, padding='valid')
	x7_face = conv_block(x6_face, 256, 1, 1)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x = conv_block(input_audio, 32)
	x = conv_block(x, 32)
	x = conv_block(x, 32)

	x = conv_block(x, 64, strides=3)	#27X9
	x = conv_block(x, 64)
	x = conv_block(x, 64)

	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = conv_block(x, 128)
	x = conv_block(x, 128)

	x = conv_block(x, 256, strides=3)	#3X3
	x = conv_block(x, 256)
	x = conv_block(x, 256)

	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, 1)

	embedding = Concatenate(axis=3)([x7_face, x])

	############# decoder
	x = conv_block(embedding, 512, 1)
	x = conv_t_block(embedding, 512, 3, 3)# 3x3
	x = Concatenate(axis=3) ([x5_face, x]) 

	x = conv_t_block(x, 512) #6x6
	x = Concatenate(axis=3) ([x4_face, x])

	x = conv_t_block(x, 256) #12x12
	x = Concatenate(axis=3) ([x3_face, x])

	x = conv_t_block(x, 128) #24x24
	x = Concatenate(axis=3) ([x2_face, x])

	x = conv_t_block(x, 64) #48x48
	x = Concatenate(axis=3) ([x1_face, x])

	x = conv_t_block(x, 32) #96x96
	x = Concatenate(axis=3) ([identity_mapping, x])

	x = conv_block(x, 16) #96x96
	x = conv_block(x, 16) #96x96
	x = Conv2D(filters=3, kernel_size=1, strides=1, padding="same") (x)
	prediction = Activation("sigmoid", name="prediction")(x)
	
	model = Model(inputs=[input_face, input_audio], outputs=prediction)
	model.summary()		
	model.compile(loss='mae', optimizer=(Adam(lr=args.lr) if hasattr(args, 'lr') else 'adam')) 
	
	return model

def create_model_residual(args, mel_step_size):
	def residual_block(inp, num_filters):
		x = conv_block(inp, num_filters)
		x = conv_block(x, num_filters)

		x = Add()([x, inp])
		x = Activation('relu') (x)

		return x

	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face")

	identity_mapping = conv_block(input_face, 32, kernel_size=7) # 96x96

	x1_face = conv_block(identity_mapping, 64, kernel_size=5, strides=2) # 48x48
	x1_face = residual_block(x1_face, 64)
	x1_face = residual_block(x1_face, 64)

	x2_face = conv_block(x1_face, 128, 3, 2) # 24x24
	x2_face = residual_block(x2_face, 128)
	x2_face = residual_block(x2_face, 128)
	x2_face = residual_block(x2_face, 128)

	x3_face = conv_block(x2_face, 256, 3, 2) #12x12
	x3_face = residual_block(x3_face, 256)
	x3_face = residual_block(x3_face, 256)

	x4_face = conv_block(x3_face, 512, 3, 2) #6x6
	x4_face = residual_block(x4_face, 512)
	x4_face = residual_block(x4_face, 512)

	x5_face = conv_block(x4_face, 512, 3, 2) #3x3
	x6_face = conv_block(x5_face, 512, 3, 1, padding='valid')
	x7_face = conv_block(x6_face, 512, 1, 1)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")
	x = conv_block(input_audio, 32)
	x = residual_block(x, 32)
	x = residual_block(x, 32)

	x = conv_block(x, 64, strides=3)	#27X9
	x = residual_block(x, 64)
	x = residual_block(x, 64)

	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = residual_block(x, 128)
	x = residual_block(x, 128)

	x = conv_block(x, 256, strides=3)	#3X3
	x = residual_block(x, 256)
	x = residual_block(x, 256)

	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, 1)
	embedding = Concatenate(axis=3)([x7_face, x])

	############# decoder
	x = conv_t_block(embedding, 512, 3, 3)# 3x3
	x = Concatenate(axis=3) ([x5_face, x]) 

	x = conv_t_block(x, 512) #6x6
	x = residual_block(x, 512)
	x = residual_block(x, 512)
	x = Concatenate(axis=3) ([x4_face, x])

	x = conv_t_block(x, 256) #12x12
	x = residual_block(x, 256)
	x = residual_block(x, 256)
	x = Concatenate(axis=3) ([x3_face, x])

	x = conv_t_block(x, 128) #24x24
	x = residual_block(x, 128)
	x = residual_block(x, 128)
	x = Concatenate(axis=3) ([x2_face, x])

	x = conv_t_block(x, 64) #48x48
	x = residual_block(x, 64)
	x = residual_block(x, 64)
	x = Concatenate(axis=3) ([x1_face, x])

	x = conv_t_block(x, 32) #96x96
	x = Concatenate(axis=3) ([identity_mapping, x])
	x = conv_block(x, 16) #96x96
	x = conv_block(x, 16) #96x96

	x = Conv2D(filters=3, kernel_size=1, strides=1, padding="same") (x)
	prediction = Activation("sigmoid", name="prediction")(x)
	
	model = Model(inputs=[input_face, input_audio], outputs=prediction)
	model.summary()		
	
	model.compile(loss='mae', optimizer=(Adam(learning_rate=args.lr) if hasattr(args, 'lr') else 'adam')) 
	return model


def create_combined_model(generator, discriminator, args, mel_step_size):

	gan = GAN(discriminator=discriminator, generator=generator)
	gan.compile(
		d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
		g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
		g_loss_fn=keras.losses.mean_absolute_error,
		d_loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
	)
	
	return gan

if __name__ == '__main__':
	model = create_model_residual()
	#plot_model(model, to_file='model.png', show_shapes=True)