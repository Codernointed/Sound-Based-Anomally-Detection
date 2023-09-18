import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tensorflow as tf
import sounddevice as sd
import numpy as np
import serial


interpreter = tf.lite.Interpreter(model_path='soundclassifier_with_metadata.tflite')
interpreter.allocate_tensors()


with open('labels.txt', 'r') as labels_file:
    labels = labels_file.read().splitlines()

#Tkinter window
root = tk.Tk()
root.title("Sound Anomally Detection")


# arduino = serial.Serial('COM5', 9600) 


fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], lw=2)
ax.set_title("Real-time Waveform")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")  
ax.set_xlim(0, 2)
ax.set_ylim(-0.2, 0.2) 
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

processing = False

consecutive_loops = 0


def process_audio():
    global processing
    global consecutive_loops
    processing = True

    duration = 2 
    samplerate = 11008  
    channels = 2  
    
    frames = int(duration * samplerate)
    while processing:
        audio_input = sd.rec(frames=frames, samplerate=samplerate, channels=channels)
        sd.wait() 
        
        # Preprocess audio input
        audio_input = audio_input.flatten() 
        input_data = np.array(audio_input, dtype=np.float32)
        
        
        input_data = input_data.reshape(1, -1)
        
        # Run on the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Map prediction to labels
        predicted_label_index = np.argmax(output_data)
        predicted_label = labels[predicted_label_index]
        
        # Update Tkinter display with predictions
        prediction_label.config(text=f"Predicted Label: {predicted_label}")
        root.update()
        
        # Check loops and change color
        if predicted_label in ["0 Anomaly Sound", "1 Machine Stopped"]:
            consecutive_loops += 1
            if consecutive_loops >= 2:
                prediction_label.config(fg='red')
                # arduino.write(b"H")
        else:
            consecutive_loops = 0
            prediction_label.config(fg='black')
            # arduino.write(b"L")
        
        # if predicted_label in ["1 Background Noise"]:
        #     arduino.write(b"H")
        # else:
        #     arduino.write(b"L")
        
        # Update plot
        line.set_data(np.arange(len(audio_input)) / samplerate, audio_input)
        ax.relim() 
        ax.autoscale_view()
        canvas.draw()
    
    prediction_label.config(text="Predicted Label: None", fg='black')
    # arduino.write(b'L')  
def stop_processing():
    global processing
    processing = False

    
    # arduino.close()

capture_button = tk.Button(root, text="Start Capturing",command=process_audio)
capture_button.pack()

stop_button = tk.Button(root, text="Stop Capturing", command=stop_processing)
stop_button.pack()

# Label to display predictions
prediction_label = tk.Label(root, text="Predicted Label: None", font=("Roboto", 20))
prediction_label.pack()

root.mainloop()