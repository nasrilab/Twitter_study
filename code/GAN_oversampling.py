#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:19:36 2023

Lyme paper- Oversampling tweets
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Assuming you have already prepared data in the form of numpy arrays:
# major_class_tweets, minor_class_tweets, minor_class_labels

# Define the GAN components
def build_generator(latent_dim, tweet_dim):
    # Define generator architecture
    pass

def build_discriminator(tweet_dim):
    # Define discriminator architecture
    pass

# GAN parameters
latent_dim = 100
tweet_dim = len(minor_class_tweets[0])

# Build and compile the generator and discriminator
generator = build_generator(latent_dim, tweet_dim)
discriminator = build_discriminator(tweet_dim)

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Combine generator and discriminator to form GAN
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
fake_tweet = generator(gan_input)
gan_output = discriminator(fake_tweet)
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
epochs = 10000
batch_size = 32

for epoch in range(epochs):
    # Select a random batch of tweets from the minor class
    batch_tweets = np.random.choice(minor_class_tweets, size=batch_size, replace=False)
    batch_labels = np.ones(batch_size)  # Labels for the synthetic tweets

    # Generate noise as input to the generator
    noise = np.random.normal(0, 1, size=[batch_size, latent_dim])

    # Generate synthetic tweets using the generator
    generated_tweets = generator.predict(noise)

    # Combine real and synthetic tweets and their labels
    X = np.concatenate([batch_tweets, generated_tweets])
    y = np.concatenate([batch_labels, np.zeros(batch_size)])

    # Train the discriminator on the combined dataset
    d_loss = discriminator.train_on_batch(X, y)

    # Train the generator
    noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
    valid_labels = np.ones(batch_size)  # Labels for the generator to learn
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Optionally print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

# After training, the generator is capable of generating synthetic tweets for the minor class
# You can use it to generate as many synthetic tweets as needed.

# For example:
num_synthetic_tweets = 8000
noise = np.random.normal(0, 1, size=[num_synthetic_tweets, latent_dim])
synthetic_tweets = generator.predict(noise)