#Written by Tim Aris
import os
import time
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv1D
from keras.optimizers import Adam

from DataGen import *

#Try a RNN
#convert to 11015 and see how that sounds
#once dim is sufficiently reduced, run through autoencoder
#maybe we can still run w/ 44k, autoencoders are for dim reduction, 
    #after all
    
#validate should be on samples, not whole songs
    
def testSong(model, song, doodle, window):
    #print("    Accuracy for ", song.title, " and ", doodle.title)
    if doodle.song != song:
        Ytest = [1, 0]
    else:
        Ytest = [0, 1]
    length = min(len(song.signal), len(doodle.signal)) // window
    predictions = []
    correct = 0
    for i in range(length):
        section = song.signal[i*window:(i+1)*window] + doodle.signal[i*window:(i+1)*window]
        section = section.reshape(1, window, 1)
        pred = model.predict(section)[0]
        predictions.append(pred)
        if (pred[0] < pred[1]) == (Ytest[0] < Ytest[1]):
            #print(1)
            correct += 1
        #else:
            #print(0)
    #print("    ", correct * 100 / length, "% match.")
    correct = correct * 100 / length
    if doodle.song != song:
        correct = 100 - correct
    correct = round(correct, 4)
    print("    ", correct, "% of song segments classified correctly for song: ", song.title, " and doodle: ", doodle.title)
    if correct > 50:
        return 1
    else: 
        return 0

def validate(model, Xv, Yv):
    pred = model.predict(Xv)
    correct = 0
    for i in range(len(Xv)):
        if (pred[i][0] <= pred[i][1]) == (Yv[i][0] <= Yv[i][1]):
            correct += 1
    print("Songs classified correctly = ", correct, " / ", len(Xv), " = ", correct/len(Xv))
    return correct / len(Xv)    

def everyCombination(model, songs, doodles, window):
    correct = 0
    n = 0
    for i in range(len(songs)):
        for j in range(len(doodles)):
            correct += testSong(model, songs[i], doodles[j], window)
            n += 1
    print("Total Correct = ", correct, "/", n,"=", correct/n)
    return correct / n

secs = 0.2
init()
#Xt, Yt = dataGen(10, secs)
#game(songs, doodles)

window = 44100
LOL = int(secs * window)
input_shape = (LOL, 1)


model = Sequential()
model.add(Conv1D(3, 1,
                 activation='relu',
                 input_shape=input_shape))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='sgd')

print("Model compiled")

songs = getSongs()
doodles = getTestDoodles()
valDoodles = getValDoodles()
p = []

#TODO stop when all in high 90s
for i in range(25):
    print("Epoch ", i)
    Xt, Yt = dataGen(100, secs, 'train')
    Xv, Yv = dataGen(100, secs, 'val')
    model.fit(Xt, Yt,
                 epochs = 1,
                 verbose=2, 
                 validation_data=((Xv, Yv)))
    
    p.append(everyCombination(model, songs, valDoodles, LOL))
"""
print("Output: ")
print(model.predict(Xv))
"""
plt.plot(p)
plt.show()




#test = testSong(model, songs[0], doodles[0], window * secs)
songs, doodles = otherDataGen()
everyCombination(model, songs, doodles, LOL)


"""
again = "y"
while again[0] == 'y':
    fuel()
    again = input("Run again? (y/n): ")
print("Goodbye")
"""
