# Drum-Doodle-Classifier
Nevermind, it never actually worked

Classifies doodles from a drum pad based on whether or not they match a song. 

How it works is that you put some .wav songs in a folder called "Songs", then record yourself 
drumming on a drum pad while listening to each one (these will be called 'doodles'), and put 
those .wav files in a folder called "Doodles". Each doodle should start with the name of the 
corresponding song. 

Then, run Drum_Doodle_Classifier.py and it will tell you how well the model did. 

Currently, it uses a very simple neural network consisting of a 1D convolutional layer, followed
by 2 dense layers. It's accuracy for song-snippets can reach high 90s. 
