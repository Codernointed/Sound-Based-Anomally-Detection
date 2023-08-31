import os
import numpy as np
import tkinter as tk
import pyaudio
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt


model = tf.keras.models.load_model('audio_classification_model.h5')


def preprocess_audio_chunk(audio_chunk):
   
    audio_chunk = audio_chunk.mean(axis=1) 
    
    
    target_sample_rate = 16000
    resampled_chunk = tfio.audio.resample(audio_chunk, rate_in=RATE, rate_out=target_sample_rate)
    
    
    target_length = 48000  
    if len(resampled_chunk) < target_length:
        resampled_chunk = np.concatenate((resampled_chunk, np.zeros(target_length - len(resampled_chunk))))
    else:
        resampled_chunk = resampled_chunk[:target_length]
    
    spectrogram = tf.signal.stft(resampled_chunk, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    
    return spectrogram

# Function to update GUI
def update_gui(predictions):
    if predictions[0] > 0.5:
        result_label.config(text="Predicted: Normal Sound")
    else:
        result_label.config(text="Predicted: Anomally Sound")

# Real-time audio processing loop
def audio_stream_callback(in_data, frame_count, time_info, status):
    audio_chunk = np.frombuffer(in_data, dtype=np.float32).reshape(-1, 2)
    preprocessed_chunk = preprocess_audio_chunk(audio_chunk)
    predictions = model.predict(np.expand_dims(preprocessed_chunk, axis=0))
    update_gui(predictions)
    return (in_data, pyaudio.paContinue)

# Initialize PyAudio
p = pyaudio.PyAudio()
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
CHUNK = 1024

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_stream_callback)

# Tkinter GUI
root = tk.Tk()
root.title("Real-Time Sound Classification")

result_label = tk.Label(root, text="Predicted: ", font=("Helvetica", 16))
result_label.pack(pady=20)

def close_window():
    stream.stop_stream()
    stream.close()
    p.terminate()
    root.destroy()

close_button = tk.Button(root, text="Close", command=close_window, font=("Helvetica", 12))
close_button.pack()

root.mainloop()
