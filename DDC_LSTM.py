#Written by Tim Aris
import os
import time
import math
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout
from keras.optimizers import Adam

import DataGen as dg

#TODO:
#CRNN
#game
#determninistic data gen
#Autoencoder - try literally not shrinking the data size
    #even that didn't work, more epochs?
    
    
def testSong(model, song, doodle, window):
    #print("    Accuracy for ", song.title, " and ", doodle.title)
    if doodle.song != song:
        Ytest = [1, 0]
    else:
        Ytest = [0, 1]
    print('Correct: ', Ytest)
    length = min(len(song.signal), len(doodle.signal)) // window
    predictions = []
    correct = 0
    for i in range(length):
        section = song.signal[i*window:(i+1)*window] + doodle.signal[i*window:(i+1)*window]
        section = section.reshape(1, window, 1)
        pred = model.predict(section)[0]
        print('Pred', i, ':', pred, end = '')
        predictions.append(pred)
        if (pred[0] < pred[1]) == (Ytest[0] < Ytest[1]):
            print("  true")
            correct += 1
        print()
        #else:
            #print(0)
    #print("    ", correct * 100 / length, "% match.")
    correct = correct * 100 / length
    if doodle.song != song:
        correct = 100 - correct    #THIS IS THE PROBLEM YOU FUCKING HAAACK!
    correct = round(correct, 4)
    print("    ", correct, "% of song segments classified correctly for song: ", song.title, " and doodle: ", doodle.title)
    if correct > 50:
        return 1
    else: 
        return 0
    
def printTime(ttime):
    hours = int(math.floor(ttime/3600.0))
    mins = int(math.floor((ttime-hours*3600.0)/60.0))
    secs = int(ttime-hours*3600.0-mins*60.0)
    if hours > 0:
        print( hours, "hours,",end = "")
    if mins > 0:
        print( mins, "minutes,",end = "")
    print( secs,"seconds")

def validate(model, Xv, Yv):
    pred = model.predict(Xv)
    correct = 0
    for i in range(len(Xv)):
        if (pred[i][0] == Yv[i][0]):
            correct += 1
    print("Segments classified correctly = ", correct, " / ", len(Xv), " = ", correct/len(Xv))
    return correct / len(Xv) 

def validate2(model, Xv, Yv):
    pred = model.predict(Xv)
    correct = 0
    for i in range(len(Xv)):
        if (pred[i][0] <= pred[i][1]) == (Yv[i][0] <= Yv[i][1]):
            correct += 1
    print("Segments classified correctly = ", correct, " / ", len(Xv), " = ", correct/len(Xv))
    return correct / len(Xv) 

def validate3(model, Xv, Yv):
    pred = model.predict(Xv)
    correct = 0
    for i in range(len(Xv)):
        if (pred[i] <= 0.5) == (Yv[i] <= 0.5):
            correct += 1
    print("Segments classified correctly = ", correct, " / ", len(Xv), " = ", correct/len(Xv))
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

def predictSong(model, song, doodles, window):
    herpDerp = 0
    length = len(song.signal) // window
    predictions = []
    
    for i in range(length):
        best = 0
        chosenDoodle = None
        for doodle in doodles:
            try:
                section = song.signal[i*window:(i+1)*window] + doodle.signal[i*window:(i+1)*window]
                section = section.reshape(1, window, 1)
                pred = model.predict(section)[0]
                if best < pred[1]:
                    best = pred[1]
                    chosenDoodle = doodle
                s = ''
                if doodle.title[1:5] == song.title[1:5]:
                    s = "THIS IS A MATCH"
                #print("prediction = ", pred, s)
            except IndexError:
                herpDerp += 1
                #print("doodle to short")
            except ValueError:
                herpDerp += 1
                #print("doodle to short")
            
        #print("Best doodle for this section was ", chosenDoodle.title)
        predictions.append(chosenDoodle)
    
    """
    for doodle in doodles:
        title = doodle.title
        print(title, " = ", predictions.count(title))
    """
    
    return max(set(predictions), key=predictions.count)

def correctSongs(model, songs, doodles, window):
    correct = 0
    total = len(songs)
    for song in songs:  
        predictedDoodle = predictSong(model, song, doodles, window)
        if (predictedDoodle.song == song):
            correct += 1
        
    print(correct, "out of", total, "songs correctly classified")
    return correct / total
        
    


dg.init()
#Xt, Yt = dataGen(10, secs)


#times = [0]
#frequencies = [0]
secs = 5 
freq = 5512
window = int(secs * freq)
input_shape = (window, 1)


model = Sequential()
model.add(Conv1D(256, 4, strides=2, input_shape=(window,1)))
model.add(MaxPooling1D())
model.add(Conv1D(256, 4, strides=2))
model.add(MaxPooling1D())
model.add(Conv1D(256, 4, strides=2))
model.add(MaxPooling1D())
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(16))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Dropout(0.1))
model.compile(loss='categorical_crossentropy', optimizer='sgd')
print("Model compiled")
model.summary()


start = time.time()
p = []
for i in range(20):
    print("Epoch ", i)
    Xt, Yt = dg.dataGen(100, secs, 'train')
    #Yt = Yt[:,0]
    Xv, Yv = dg.dataGen(100, secs, 'val')
    #Yv = Yv[:,0]
    model.fit(Xt, Yt,
                 epochs = 1,
                 verbose=2, 
                 validation_data=((Xv, Yv)))
    
    #correctSongs(model, songs, doodles, freq)
    p.append(validate2(model, Xv, Yv))

print("Training Time:", end='')
printTime(time.time() - start)

plt.plot(p)
plt.show()
plt.close()
    
    #times.append(time.time() - start)
    #frequencies.append(window)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Training time (seconds)")
plt.plot(frequencies, times, 'bo-')
plt.show()
plt.close()
#model.save('LSTM.h5')
print('Model Saved')

gameStart = input("Would you like to play a game? ")
if gameStart[0] == 'y':
    humanCorrect = 0
    aiCorrect = 0
    total = 0
    songs = dg.getSongs()
    doodles = dg.getTestDoodles()
    again = 'yepperuni'
    while total < 20:
        total += 1
        if np.random.randint(2):
            match = True
            n = np.random.randint(len(songs))
            song = songs[n]
            doodle = doodles[n]
        else:
            n = np.random.randint(len(songs))
            m = np.random.randint(len(songs))
            match = n == m
            song = songs[n]
            doodle = doodles[m]
        end = min(len(song.signal), len(doodle.signal)) - window
        start = np.random.randint(end)
        problem = song.signal[start:start+window] + doodle.signal[start:start+window]*4
        wav.write("Example"+str(total)+".wav", 5512, problem)
        a = input("Did that match? ")
        if (a[0] == 'y') == match:
            humanCorrect += 1
            print('Correct!')
        else:
            print('Nope!')
            
        problem = problem.reshape(1, window, 1)
        pred = model.predict(problem)[0]
        print("Prediction = ", pred)
        if (pred[0] < pred[1]) == match:
            aiCorrect += 1
            print("The AI was Correct!")
        else:
            print("The AI was wrong!")
        
        print("The song:", song.title, ", the doodle:", doodle.song.title)
        print('Currect Score:')
        print('Human: ', humanCorrect, '/', total)
        print('Model: ', aiCorrect, '/', total)
        print()
        #again = input('Again? ')
        
    print("Totals:")
    print('Human: ', humanCorrect, '/', total, ' = ', round(humanCorrect/total*100), '%')
    print('Model: ', aiCorrect, '/', total, ' = ', round(aiCorrect/total*100), '%')

#test = testSong(model, songs[0], doodles[0], freq * secs)
#songs, doodles = otherDataGen()
#everyCombination(model, songs, doodles, window)
#https://github.com/Timjaris/Drum-Doodle-Classifier.git
#10 epochs: 2 hours,41 minutes,53 seconds
"""
1: n
2: n
3: n
4: y
5: y
6: y
7: n
8: y
9: y
10: y
11: y
12: y
13: y
14: n
15: y
16: n
17: n
18: n
19: y
20: y
"""
