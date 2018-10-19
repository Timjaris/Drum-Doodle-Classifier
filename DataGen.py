#Written by Tim Aris
import os
import time
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import keras

songs = []
trDoodles = []
valDoodles= []
tsDoodles = []

#TODO make versus mode against trained model
def game(songs, doodles):
    s = input("How many seconds?")
    correct = 0
    total = 0
    again = "yup"
    while isInt(s) and again[0] == 'y':
        seconds = int(s)
        if np.random.randint(2):
            n = np.random.randint(len(songs))
            song = songs[n].signal
            doodle = doodles[n].signal
            end = min(len(song), len(doodle)) - seconds * 44100
            start = np.random.randint(end)
            
            problem = song[start:start+seconds*44100] + doodle[start:start+seconds*44100]*4
            plt.plot(problem)
            plt.show()
            wav.write("Problem.wav", 44100, problem)
            
            a = input("Did that match? ")
            if (a[0] == 'y'):
                correct = correct + 1
                print("Correct!")
            total = total + 1
        else:
            n = np.random.randint(len(songs))
            m = np.random.randint(len(songs))
            song = songs[n].signal
            doodle = doodles[m].signal
            end = min(len(song), len(doodle)) - seconds * 44100
            start = np.random.randint(end)
        
            problem = song[start:start+seconds*44100] + doodle[start:start+seconds*44100]*4
            wav.write("Problem.wav", 44100, problem)
            
            a = input("Did that match? ")
            if (a[0] == 'y') == (n==m):
                correct = correct + 1
                print("Correct!")
            total = total + 1
        
        again = input("Again?")
    
    print("Total accuracy: ", correct, " / ", total, " = ", correct/total)

class song():
    def __init__(self, title, path, signal):
        self.title = title
        self.path = path
        self.signal = signal
    
class doodle():
    def __init__(self, title, path, signal, song):
        self.title = title
        self.path = path
        self.signal = signal
        self.song = song

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def getSongs():
    return songs

def getDoodles():
    return trDoodles

def getTestDoodles():
    return tsDoodles

def getValDoodles():
    return valDoodles
    

def init():
    songDir = os.getcwd() + '\\Songs'
    doodleDir = os.getcwd() + '\\Doodles'
    #songs = []
    #doodles = []
    for filename in os.listdir(songDir):
        path = os.path.join(songDir, filename)
        (fs, signal) = wav.read(path)
        #print("fs = ", fs)
        signal = signal[:,0] 
        top = max(signal)
        signal = [i/top for i in signal]  
        signal = np.array(signal)
        s = song(filename, path, signal)
        songs.append(s)
        
    for filename in os.listdir(doodleDir):
        path = os.path.join(doodleDir, filename)
        (fs, signal) = wav.read(path)
        signal = signal[:,0] 
        
        #this takes a while, only use if necessary
        top = max(signal)
        signal = [i/top for i in signal]
        signal = np.array(signal)
        targetSong = None
        for sing in songs:
            if filename[0:len(filename)-5] == sing.title[0:len(sing.title)-4]:
                targetSong = sing
        d = doodle(filename, path, signal, targetSong)
        l = len(filename)
        if filename[l-5] == '1':
            trDoodles.append(d)
        elif filename[l-5] == '2':
            valDoodles.append(d)
        else:
            tsDoodles.append(d)
            
def dataGen(N, seconds, set): #timestep? 2d or 1d conv    
    data = []  
    targets = []
    for i in range(N):
        n = np.random.randint(len(songs))
        if set == 'train':
            Doodle = trDoodles[n].signal
        elif set == 'val':
            Doodle = valDoodles[n].signal
        else:
            Doodle = tsDoodles[n].signal
        
        if np.random.randint(2):
            #Correct song
            Song = songs[n].signal
            end = min(len(Song), len(Doodle)) - seconds * 44100
            start = np.random.randint(end)
            
            problem = Song[start:start+int(seconds*44100)] + Doodle[start:start+int(seconds*44100)]*2
            data.append(problem)
            targets.append([0, 1]) #Boolean?
        else:
            #Random song, will probably be wrong
            m = np.random.randint(len(songs))
            Song = songs[m].signal
            end = min(len(Song), len(Doodle)) - seconds * 44100
            start = np.random.randint(end)
        
            problem = Song[start:start+int(seconds*44100)] + Doodle[start:start+int(seconds*44100)]*2
            data.append(problem)
            target = [0,0]
            target[n==m] = 1
            targets.append(target)
            
    
    
    return np.array(data).reshape((N, int(seconds*44100), 1)), np.array(targets)#.reshape(N, 1)
    
#Songs unlike the training data by various degrees
def otherDataGen():
    songDir = os.getcwd() + '\\Other\\Songs'
    doodleDir = os.getcwd() + '\\Other\\Doodles'
    songs = []
    doodles = []
    for filename in os.listdir(songDir):
        path = os.path.join(songDir, filename)
        (fs, signal) = wav.read(path)
        signal = signal[:,0] 
        top = max(signal)
        signal = [i/top for i in signal]  
        signal = np.array(signal)
        s = song(filename, path, signal)
        songs.append(s)
        
    for filename in os.listdir(doodleDir):
        path = os.path.join(doodleDir, filename)
        (fs, signal) = wav.read(path)
        signal = signal[:,0] 
        
        #this takes a while, only use if necessary
        top = max(signal)
        signal = [i/top for i in signal]
        signal = np.array(signal)
        targetSong = None
        for sing in songs:
            l = len(sing.title)-4
            if filename[0:l] == sing.title[0:l]:
                targetSong = sing
        d = doodle(filename, path, signal, targetSong)
        doodles.append(d)
        
    return songs, doodles
    
"""
testSong = os.path.join(songDir, "Another Medium.wav")
testDoodle = os.path.join(doodleDir, "Another Medium1.wav")

(fs, signal1) = wav.read(testSong)
signal1 = signal1[:,0]                #44kHz

(fs, signal2) = wav.read(testDoodle)   
signal2 = signal2[:,0]                #44kHz

#test = signal1 + signal2
def output(song, doodle, start, stop):  
    plt.plot(song[start:stop])
    plt.title = "Song"
    plt.show()
    plt.figure(2)
    plt.title = "Doodle"
    plt.plot(signal2[start:stop])
    plt.show()
    
start = 0
stop = min(len(signal1), len(signal2))
#output(signal1, signal2, 0, min(len(signal1), len(signal2)))

test = signal1[start:stop] + signal2[start:stop]
wav.write("test.wav", 44100, test)
#plt.plot(test)
#plt.show()
"""
