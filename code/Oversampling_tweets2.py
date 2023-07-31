#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Oversampling tweets-Lyme paper

import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the tweet data
tweets_df = pd.read_csv('tweets.csv')

# Define the minority class label
minor_class = 1

# Split the data into minority and majority classes
minor_tweets = tweets_df[tweets_df['label'] == minor_class]
major_tweets = tweets_df[tweets_df['label'] != minor_class]

# Split the minority class data into training and testing sets
minor_train, minor_test = train_test_split(minor_tweets, test_size=0.2, random_state=42)

# Define the dimensionality of the noise vector
noise_dim = 100

# Define the generator model
def build_generator():
    generator = Sequential()

    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dense(len(minor_tweets.columns), activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return generator

# Define the discriminator model
def build_discriminator():
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=len(minor_tweets.columns)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return discriminator

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = Input(shape=(noise_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    return gan

# Define a function to train the GAN
def train_gan(generator, discriminator, gan, X_train, epochs=20000, batch_size=128):
    for i in range(epochs):
        # Generate noise vector
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))

        # Generate fake tweets
        fake_tweets = generator.predict(noise)

        # Select a random batch of real tweets
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_tweets = X_train[idx]

        # Train the discriminator on the real and fake tweets
        discriminator_loss_real = discriminator.train_on_batch(real_tweets, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_tweets, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

        # Train the generator to trick the discriminator
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print the progress
        if i % 1000 == 0:
            print(f'Epoch: {i}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}')

# Preprocess the tweet data
tweets_df = tweets_df.astype('float32')
X_train = major_tweets.values

# Build the generator, discriminator, and GAN models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train the GAN on the minority class data
minor_train = minor_train.astype('float32')
train_gan(generator, discriminator, gan, minor_train.values)

# Generate fake tweets to oversample the minority class
noise = np.random.normal(0, 1, size=(minor_test.shape[0], noise_dim))
fake_tweets = generator.predict(noise)
oversampled_minor = pd.DataFrame(fake_tweets, columns=minor_test.columns)

# Append the oversampled minority class data to the original data
oversampled_tweets = pd.concat([tweets_df, oversampled_minor], axis=0)

# Shuffle the data
oversampled_tweets = oversampled_tweets.sample(frac=1).reset_index(drop=True)

# Save the oversampled data to a CSV file
oversampled_tweets.to_csv('oversampled_tweets.csv', index=False)
