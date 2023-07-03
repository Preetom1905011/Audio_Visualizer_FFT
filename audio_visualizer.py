import librosa
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import pytube
import pydub
import os

def get_decibel(target_time, freq):
    # print("-->", spectrogram[int(freq * frequencies_index_ratio)][int(target_time * time_index_ratio)])
    return spectrogram[int(freq * frequencies_index_ratio)][int(target_time * time_index_ratio)]

def load_youtube_audio(url):

    youtube = pytube.YouTube(url)
    audio_stream = youtube.streams.filter(only_audio=True).first()
    audio_path = audio_stream.download()

    output_path = os.path.splitext(audio_path)[0] + ".wav"

    audio = pydub.AudioSegment.from_file(audio_path)
    audio.export(output_path, format='wav')

    os.remove(audio_path) 

    return output_path


youtube_url = "https://www.youtube.com/watch?v=pyi0ZfuIIvo"

file_path = load_youtube_audio(youtube_url)
time_series, sample_rate = librosa.load(file_path)  # getting information from the file

# getting a matrix which contains amplitude values according to frequency and time indexes
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048*4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)  # converting the matrix to decibel matrix

frequencies = librosa.core.fft_frequencies(n_fft=2048*4)  # getting an array of frequencies

print(frequencies)
print(stft.shape, len(time_series), sample_rate)
print(spectrogram)
# exit()

# getting an array of time periodic
times = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048*4)
print(times, len(time_series) / sample_rate)

time_index_ratio = len(times)/times[len(times) - 1]

frequencies_index_ratio = len(frequencies)/frequencies[len(frequencies)-1]

x_frequencies = np.arange(100, int(frequencies[-1] / 100) * 100, 100)
T = int(times[len(times) - 1])
print(T)
r = len(x_frequencies)
print(r)

width = 5

x = 2.5

fig, ax = plt.subplots()

x_pts = np.arange(x, x + width * r, width)
# print(r, len(x_pts))
ax.set_ylim([0, 90])


def animate(t):
    global cur_time
    ax.cla() # clear the previous image
    time_in_s = t * (interval / 1000)
    cur_time = time_in_s
    y_pts = [ 80 + get_decibel(time_in_s, f) for f in x_frequencies]
    ax.bar(x_pts, y_pts, width=width, color=(250/256, 1 - np.max(y_pts)*(3/256), 1 - np.max(y_pts)*(3/256))) # plot the line
    ax.set_xticks([x_pts[i] for i in range(0, len(x_pts), 10)], [x_frequencies[i] for i in range(0, len(x_frequencies), 10)])
    # plt.gca().invert_yaxis()
    ax.set_ylim([0, 90]) # fix the y axis
    ax.set_title("Time: " + str(np.round(time_in_s, 4)) + " seconds")

# interval in milliseconds
interval = 100
frames = int(T / (interval / 1000))
print(frames)
# exit()
pygame.mixer.init()
pygame.mixer.music.load(file_path)

cur_time = 0
anim = animation.FuncAnimation(fig, animate, frames = frames, interval = interval, blit = False, repeat=False)

pygame.mixer.music.play(0, start=cur_time)
plt.show()

os.remove(file_path)