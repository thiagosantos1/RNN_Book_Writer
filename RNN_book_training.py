#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:50:09 2018

@author: thiago
"""

# File for training the RNN

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM # RNN
from keras.layers import Dropout # used for regularization, avoiding overfitting
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# Part 1 - Data Preprocessing

# load ascii text and covert to lowercase
# to reduce the vocabulary that the network must learn.
filename = 'Dataset/shakespeare_first_51.txt'
raw_text = open(filename).read()
raw_text = raw_text.lower()



# create mapping of unique chars to integers
# We cannot model the characters directly, instead we must convert the characters to integers.

# We can do this easily by first creating a set of all of the distinct characters in the book
chars = sorted(list(set(raw_text)))
# then creating a map of each character to a unique integer.
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the dataset
n_chars = len(raw_text)
n_vocab = len(chars)
#print ("Total Characters: ", n_chars)
#print( "Total Vocab: ", n_vocab)


# prepare the dataset of input to output pairs encoded as integers

# Define the size of the memorization
seq_length = 150 # gonna remeber a sequence of 150 chars, to predict next
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
#print ("Total Patterns: ", n_patterns)

# Reshape #
# Now that we have prepared our training data we need to transform it
# so that it is suitable for use with Keras.


# reshape X to be [samples, time steps, features]
X_train = np.reshape(dataX, (n_patterns, seq_length, 1))


# Next we need to rescale the integers to the range 0-to-1 to make the patterns easier
# to learn by the LSTM network that uses the sigmoid activation function by default.
# normalize
X_train = X_train / float(n_vocab)


# one hot encode the output variable
# This is so that we can configure the network to predict the probability of each of
# the 47 different characters in the vocabulary (an easier representation) rather
# than trying to force it to predict precisely the next character.
y_train = np_utils.to_categorical(dataY)



# Part 2 - Building the RNN

# The problem is really a single character classification problem with 48 classes and as
# such is defined as optimizing the log loss (cross entropy), here using the ADAM optimization
# algorithm for speed.
# Initialising the RNN, as always
classifier = Sequential()

# Best way - Needs GPU

classifier.add(LSTM(256,  return_sequences = True,  input_shape=(X_train.shape[1], X_train.shape[2])))
classifier.add(Dropout(0.2))

classifier.add(LSTM(256,  return_sequences = True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(128))
classifier.add(Dropout(0.2))

classifier.add(Dense(y_train.shape[1], activation='softmax')) # output with 48 nodes, to predict
classifier.compile(optimizer='adam', loss='categorical_crossentropy')

"""
# Option for CPU
classifier.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
classifier.add(Dropout(0.2))
classifier.add(Dense(y_train.shape[1], activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy')
"""

# There is no test dataset. We are modeling the entire training dataset to learn the probability
# of each character in a sequence.

"""
We are not interested in the most accurate (classification accuracy) model of the training dataset.
This would be a model that predicts each character in the training dataset perfectly.
Instead we are interested in a generalization of the dataset that minimizes the chosen loss function.
We are seeking a balance between generalization and overfitting but short of memorization.
"""

"""
The network is slow to train (about 300 seconds per epoch on a GPU). Because of the slowness and
because of our optimization requirements, we will use model checkpointing to record all of the
network weights to file each time an improvement in loss is observed at the end of the epoch.
We will use the best set of weights (lowest loss) to instantiate our generative model
in the next section.
"""

# define the checkpoint
# This is because we can just reload the weights, to execute a prediction
# Saves times. no need to training again
filepath='Dataset/weights_improvement_{epoch:02d}_{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Trainig Part
# Fitting the RNN to the Training set
# epochs = number of repetitions
# batch_size = how to divide the data to be trained
    # after 128 chars, that we gonna run the backpropagation

classifier.fit(X_train, y_train, epochs=40, batch_size=64, callbacks=callbacks_list)

# After fit, we should go to the file weights and remove all lines, expect for the best one(lower loss)


### Generating text using the trained LSTM network ###

"""
load the data and define the network in exactly the same way, except the network weights are
loaded from a checkpoint file and the network does not need to be trained.
"""

