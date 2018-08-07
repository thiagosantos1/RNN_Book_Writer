#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:50:09 2018

@author: thiago
"""

# File to load the trained weights and write the book

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

"""
# Option for CPU
classifier.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
classifier.add(Dropout(0.2))
classifier.add(Dense(y_train.shape[1], activation='softmax'))
"""


### Generating text using the trained LSTM network ###

"""
load the data and define the network in exactly the same way, except the network weights are
loaded from a checkpoint file and the network does not need to be trained.
"""

# load the network weights
filepath='Dataset/weights_improvement_01_3.0288.hdf5'
classifier.load_weights(filepath)
classifier.compile(optimizer='adam', loss='categorical_crossentropy')

"""
create a reverse mapping that we can use to convert the integers back to characters so
that we can understand the predictions.
"""
int_to_char = dict((i, c) for i, c in enumerate(chars))


# Write the book --> make predictions
"""
The simplest way to use the Keras LSTM model to make predictions is to first start off with a
seed sequence as input, generate the next character then update the seed sequence to add the
generated character on the end and trim off the first character. This process is repeated for as
long as we want to predict new characters (e.g. a sequence of 1,000 characters in length).

We can pick a random input pattern as our seed sequence, then print generated characters
as we generate them.
"""

# Size of book, in chars
size_pred = 500

# pick a random seed --> with same size used for the training
# This means we gonna get some memory, and then predict next chars
# From this seed, machine tries to predict the whole pattern selected
# and after the patter is gone, it keep trying to predict on its own
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
print("Text generated:" )
for i in range(size_pred):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = classifier.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print ("\nDone.")



