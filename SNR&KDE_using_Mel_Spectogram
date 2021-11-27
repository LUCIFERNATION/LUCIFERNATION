import numpy as np
#std = sqrt(mean(abs(x - x.mean())**2)).

a = np.array([[1, 2], [3, 4]])
print (np.std(a))
print (np.std(a, axis=0))
print (np.std(a, axis=1))
#SNR
import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

Arr = [[20, 4, 7, 1, 34], [50, 12, 15, 34, 5]] 
singleChannel = Arr
try:
    singleChannel = np.sum(Arr, axis=1)
    print ("# Single channel: ")
    print (singleChannel)
except:
    # was mono after all
    pass
print ("# np.amax(singleChannel)")
print (np.amax(singleChannel))
print ("# -1 * np.amin(singleChannel)")
print (-1 * np.amin(singleChannel))
norm = singleChannel / (max(np.amax(singleChannel), -1 * np.amin(singleChannel)))
print ("# Norm")
print (norm)
snr = signaltonoise(norm, axis=0, ddof=0)
print ("# Signal to Noise Ratio: ")
print (snr)
#KDE
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

bandwidths = [0.1, 0.2, 0.3, 0.4, 0.5]
for bd in bandwidths:
    kde = KernelDensity(kernel='gaussian', bandwidth=bd).fit(X)
    scores = np.array(kde.score_samples(X))
    print ("# bandwidth : "+str(bd))
    print (scores)

#Normal heartbeat SOUND
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import numpy as np

file_source = '/Users/mac/Documents/proposal_training_heartbeat/course1/datasets/pascal/normal/101_1305030823364_B.wav'
signal, sample_rate = librosa.load(file_source)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(signal, sr=sample_rate)
plt.show()

signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
plt.figure(figsize=(14, 5))
librosa.display.waveplot(signal, sr=sample_rate)
plt.show()

n_fft = 2048
D = np.abs(librosa.stft(signal[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
plt.plot(D);
#Now let’s take the complete signal, separate it to time windows, 
#and apply the Fourier Transform on each time window.
#The number of samples between successive frames, e.g., 
#the columns of a spectrogram. This is denoted as a positive integer hop_length

hop_length = 512
D = np.abs(librosa.stft(signal, n_fft=n_fft,  hop_length=hop_length))
librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='linear');
plt.colorbar();
#Wow can’t see much here can we? 
#It’s because most sounds humans hear are concentrated in very small frequency and amplitude ranges.
#Let’s make another small adjustment - transform both the y-axis (frequency) to log scale, 
#and the “color” axis (amplitude) to Decibels, which is kind a the log scale of amplitudes.

DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');
#Now this is what we call a Spectrogram!
#The Mel Scale
n_mels = 128
mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

plt.figure(figsize=(15, 4));
#subplot(nrows, ncols, index)
plt.subplot(1, 3, 1);
librosa.display.specshow(mel, sr=sample_rate, hop_length=hop_length, x_axis='linear');
plt.ylabel('Mel filter');
plt.colorbar();
plt.title('1. Our filter bank for converting from Hz to mels.');

plt.subplot(1, 3, 2);
mel_10 = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=10)
librosa.display.specshow(mel_10, sr=sample_rate, hop_length=hop_length, x_axis='linear');
plt.ylabel('Mel filter');
plt.colorbar();
plt.title('2. Easier to see what is happening with only 10 mels.');

plt.subplot(1, 3, 3);
idxs_to_plot = [0, 9, 49, 99, 127]
for i in idxs_to_plot:
    plt.plot(mel[i]);
plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
plt.title('3. Plotting some triangular filters separately.');

plt.tight_layout();
plt.plot(D[:, 1]);
plt.plot(mel.dot(D[:, 1]));
plt.legend(labels=['Hz', 'mel']);
plt.title('One sampled window for example, before and after converting to mel.');
#The Mel Spectrogram

S = librosa.feature.melspectrogram(signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB');
