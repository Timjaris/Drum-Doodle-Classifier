# Drum-Doodle-Classifier
The Drum Doodle Classifier (DDC) was an attempt to train a neural network to recognize whether a song snippett was paired with a drum-pad-pattern that matched it (was made while listening to the song), or not (was made listening to some other song). With every method I tried, however, the training error rate never fell below random chance (50%). Most of the models were 1D convolutional networks of varying structures, with the input being the sum of the .wav files for the song and doodle (within the window, which was usually 4 seconds). Other models used were RNNs, CRNNs, LSTMs, and a 2D CNN (This one with the spectrogram. Since the relevant data was purely rhythmic, I didn't think the frequencies mattered, but both didn't work so who knows.).
 
Possibility: 
I remain convinced that this is possible. I was able to consistently classify 19/20 to 20/20 snippets correctly Maybe I have a special advantage, because I'm the one who made the doodles, so I sent some samples to a not-musically-included friend, and he got 16/20 correct. Perhaps it just requires more than just my laptop's mere 1 GFLOPS to train. 

(Songs used were my favorites from the Undertale soundtrack; I don't think Toby Fox will mind copies of them here, since they've all been downsampled to 5012Hz, the lowest it could go while still being recognizable.)
