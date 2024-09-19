#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras import Sequential
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import copy
import itertools


# In[2]:


classList = ['dry', 'healthy', 'sick','Good Morning']
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
sequenceLength = 30


# In[3]:


def normalizeCoordinates(coords):

    baseX = 0
    baseY = 0

    for i, val in enumerate(coords):
        if i == 0:
            baseX = val[0]
            baseY = val[1]

        coords[i][0] = coords[i][0] - baseX
        coords[i][1] = coords[i][1] - baseY

    coords = list(itertools.chain.from_iterable(coords))

    maxVal = max(list(map(abs, coords)))

    def normalize_(n):
        return n / maxVal

    coords = list(map(normalize_, coords))

    return coords


# In[4]:


def skeletonExtraction(path):

    left = []
    right = []
    cap = cv2.VideoCapture(path)

    while True:
        success, img = cap.read()

        if (success == False):

            cap.release()
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        hLeft = None
        hRight = None

        if (results.multi_hand_landmarks):
            for idx, handLms in enumerate(results.multi_hand_landmarks):

                hand = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y*h)
                    hand.append([cx, cy])

                label = results.multi_handedness[idx].classification[0].label

                if (label == 'Left'):
                    hLeft = normalizeCoordinates(hand)
                elif (label == 'Right'):
                    hRight = normalizeCoordinates(hand)

        if (hLeft != None):
            left.append(hLeft)
        if (hRight != None):
            right.append(hRight)

    countLeft = len(left)
    countRight = len(right)
    windowLeft = max(countLeft/sequenceLength, 1)
    windowRight = max(countRight/sequenceLength, 1)

    finalFeatures = []

    if countLeft < sequenceLength or countRight < sequenceLength:
        return []

    for i in range(0, sequenceLength):

        finalFeatures.append(
            left[int(i * windowLeft)] + right[int(i * windowRight)])

    return np.asarray(finalFeatures)


# In[5]:


def inputProcessing():
    features = np.asarray([skeletonExtraction(select_video_file_name)])
    jsonFile = open('my.json', 'r')
    loadedModel = jsonFile.read()
    loadedModel = keras.models.model_from_json(loadedModel)
    loadedModel.load_weights('my.h5')
    loadedModel.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    output = loadedModel.predict(features)
    label = classList[np.argmax(output)]
    text_box.insert(tk.END, label)
    global word_is
    word_is=label


# In[6]:


def predict(file_path):
    jsonFile = open('my.json', 'r')
    loadedModel = jsonFile.read()
    loadedModel = keras.models.model_from_json(loadedModel)
    loadedModel.load_weights('my.h5')
    loadedModel.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    features = inputProcessing(select_video_file_name)
    output = loadedModel.predict(features)
    


# In[23]:


import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import time
# from model import predict
#os.chdir("/home/vivek/college/capstone/gui")

root = tk.Tk()
root.title("India Sign language Detector")
#root.iconbitmap("C:/Users/91790/Desktop/isl.png")
root.geometry("1920x1080")
root.configure(bg="#383838")

word_is=""
# root.configure(bg="yellow")
file_path = ''

frame = tk.Frame(height=480, width=640,bg="#383838")
frame.place(x=10, y=30)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)


filename = str(time.time())+'video.avi'
frames_per_second = 30.0
res = '480p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current caputre device
    # to the resulting resolution
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


# cap = cv2.VideoCapture(0)
# out = cv2.VideoWriter(str(time.time())+'video.avi',
#                       cv2.VideoWriter_fourcc(*'XVID'), 25, get_dims(cap, res))
recording = False
video_file_name = ""
select_video_file_name=""
# Define a function to start recording


def start_recording():
    global recording,video_file_name
    recording = True
    cap = cv2.VideoCapture(0)
    video_file_name=str(time.time())+'video.avi'
    out = cv2.VideoWriter(video_file_name,
                          cv2.VideoWriter_fourcc(*'XVID'), 30, get_dims(cap, res))
    button_start.config(text="Stop Recording",
                        command=lambda: stop_recording(cap, out))
    record_video(cap, out)
    # stop_recording
# Define a function to stop recording



def stop_recording(cap, out):
    global recording
    recording = False
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    
    button_start.config(text="Start Recording", command=start_recording)
# This function should call the Python program that records webcam video using OpenCV.


def record_video(cap, out):
    if recording:
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = image
            imgarr = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(imgarr)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        out.write(frame)
        lmain.after(10, lambda: record_video(cap, out))
        
    # cv2.waitKey(1000)


def choose_file():
    global select_video_file_name
    select_video_file_name = filedialog.askopenfilename(initialdir=".", title="Select a Video File",
                                           filetypes=(("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                                                      ("all files", "*.*")))
    display_video(select_video_file_name)


    
    
def display_video(file_path):
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        return
    h, w, channels = frame.shape
    scale_factor = 0.25  # Reduce the size by 4x
    h = int(h * scale_factor)
    w = int(w * scale_factor)
    video_canvas.config(width=w, height=h)
    video_canvas.place(x=650,y=30)
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        video_canvas.img_tk = img_tk
        video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        root.update()
        time.sleep(0.01)
        ret, frame = cap.read()
    cap.release()

video_canvas = tk.Canvas(root)

button1 = tk.Button(root, text="select video",command=choose_file,bg='#383838',fg='#ffffff')
button1.place(x=650, y=10)

button_start = tk.Button(root, text="Start Recording", command=start_recording,bg='#383838',fg='#ffffff')
button_start.place(x=10, y=10)

button_pred=tk.Button(root, text="Predict",command=inputProcessing,bg='#383838',fg='#ffffff')
button_pred.place(x=650,y=260)

text_box = tk.Text(root,height=2, width=14,bg="#141620",fg='#ffffff')
text_box.place(x=650,y=280)

#text_box1 = tk.Text(root,height=2, width=14,bg="#141414")
#text_box1.place(x=650,y=500)

#options = ["telugu", "tamil", "hindi"]
#selected_option = tk.StringVar()

#selected_option.set(options[0])

#dropdown = tk.OptionMenu(root, selected_option, *options)
#dropdown.place(x=650,y=380)

#button = tk.Button(root, text="Display Selection", command=lambda: translate(selected_option.get(),word_is))
#button.place(x=650,y=450)


root.mainloop()


# In[8]:


def translate(selected,word):
    global selected_option
    from translate import Translator
    translator= Translator(to_lang=selected)
    translation = translator.translate(word)
    text_box1.insert(tk.END, translation)


# In[9]:


from translate import Translator
translator = Translator(to_lang="hi")
translation = translator.translate("Good morning")
print(translation)


# In[ ]:




