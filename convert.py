import tensorflow as tf

# Load the TFLite model
tflite_model_path = 'soundclassifier_with_metadata.tflite'
tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
tflite_interpreter.allocate_tensors()

# Convert TFLite model to Keras model
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

def concrete_func(inputs):
    tflite_interpreter.set_tensor(input_details[0]['index'], inputs)
    tflite_interpreter.invoke()
    return [tflite_interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

# Create a dummy input shape for the model
dummy_input = tf.ones(input_details[0]['shape'][1:], dtype=tf.float32)

# Build the Keras model from the concrete function
keras_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_details[0]['shape'][1:]),
    tf.keras.layers.Lambda(concrete_func)
])

# Save the Keras model in HDF5 format
h5_model_path = 'converted_model.h5'
keras_model.save(h5_model_path)

print(f"Keras model saved in HDF5 format: {h5_model_path}")
# import sounddevice as sd
# import numpy as np
# import tensorflow as tf
# import librosa
# import sys

# # Load the TFLite model
# model = tf.lite.Interpreter(model_path='soundclassifier_with_metadata.tflite')
# model.allocate_tensors()

# def preprocess_audio(audio):
#     # Resample audio to the same sample rate used during training
#     sample_rate = 8806.4   
#     resampled_audio = librosa.resample(audio, orig_sr=sd.query_devices(None, 'input')['default_samplerate'],
#                                        target_sr=sample_rate)
    
#     # Clip or pad the audio to the required length
#     target_length = int(sample_rate * 5)  # 5 seconds as an example
#     preprocessed_audio = np.zeros(target_length)
#     preprocessed_audio[:len(resampled_audio)] = resampled_audio.reshape(-1)  # Reshape before assignment
    
#     return preprocessed_audio

# def predict_with_tflite(preprocessed_audio):
#     input_details = model.get_input_details()
#     output_details = model.get_output_details()
    
#     # Preprocess the audio and convert to the required input format
#     preprocessed_audio = np.expand_dims(preprocessed_audio, axis=0)
#     input_data = np.array(preprocessed_audio, dtype=np.float32)
    
#     # Set input tensor
#     model.set_tensor(input_details[0]['index'], input_data)
#     model.invoke()
    
#     # Get the output tensor
#     output_data = model.get_tensor(output_details[0]['index'])
#     print(output_data)
#     return np.max(output_data)

# print("Press Enter to start recording or 'q' to quit...")
# while True:
#     key = input()
#     if key == 'q':
#         break
#     print("Recording...")
    
#     duration = 5  
#     sample_rate = 8806.4  
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
#     sd.wait()

#     preprocessed_audio = preprocess_audio(audio)
#     prediction = predict_with_tflite(preprocessed_audio)

#     print(prediction)
#     if prediction > 0.5:
#         result = "Normal Sound"
#     else:
#         result = "Anomaly or Background Noise"

#     print(f"Prediction: {result}")
#     sys.exit()
# import tkinter as tk
# import tensorflow as tf
# import sounddevice as sd
# import numpy as np


# interpreter = tf.lite.Interpreter(model_path='soundclassifier_with_metadata.tflite')
# interpreter.allocate_tensors()

# # Load labels from labels.txt
# with open('labels.txt', 'r') as labels_file:
#     labels = labels_file.read().splitlines()

# # Create Tkinter window
# root = tk.Tk()
# root.title("Sound Classification Visualization")

# # Variable to track whether processing is ongoing
# processing = False

# # Function to process audio and update display
# def process_audio():
#     global processing
#     processing = True

#     duration = 2 
#     samplerate = 22016  
#     channels = 1  
    
#     frames = int(duration * samplerate)
#     while processing:
#         audio_input = sd.rec(frames=frames, samplerate=samplerate, channels=channels)
#         sd.wait() 
        
#         # Preprocess audio input
#         audio_input = audio_input.flatten() 
#         input_data = np.array(audio_input, dtype=np.float32)
        
#         # Reshape input data to match model's expected shape
#         input_data = input_data.reshape(1, -1)
        
#         # Run inference on the model
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
        
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
        
#         # Map prediction to labels
#         predicted_label_index = np.argmax(output_data)
#         predicted_label = labels[predicted_label_index]
        
        
#         prediction_label.config(text=f"Predicted Label: {predicted_label}")
#         root.update()  
        
#     prediction_label.config(text="Predicted Label: None") 


# def stop_processing():
#     global processing
#     processing = False


# capture_button = tk.Button(root, text="Start Capturing", command=process_audio)
# capture_button.pack()


# stop_button = tk.Button(root, text="Stop Capturing", command=stop_processing)
# stop_button.pack()

# # Label to display predictions
# prediction_label = tk.Label(root, text="Predicted Label: None")
# prediction_label.pack()

# root.mainloop()