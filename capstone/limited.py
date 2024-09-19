# %%
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras import Sequential
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import copy
import itertools

# %%
dataset = "D:\isl_projects\datasets"

classList = ['dry', 'healthy', 'sick']
print(len(classList))

# %%
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
sequenceLength = 30


# %%
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


# %%
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


# %%
def createDataset():

    features = []
    labels = []
    paths = []

    for index, name in enumerate(classList):
        filesList = os.listdir(os.path.join(dataset, name))

        for i in filesList:

            path = os.path.join(dataset, name, i)

            extractedFeatures = skeletonExtraction(path)

            if (len(extractedFeatures) == sequenceLength):
                features.append(extractedFeatures)
                labels.append(index)
                paths.append(path)

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, paths


# %%
features, labels, paths = createDataset()

# %%

# %%
featuresFile = open('features', 'rb')
labelsFile = open('labels', 'rb')
pathsFile = open('paths', 'rb')
features = pickle.load(featuresFile)
labels = pickle.load(labelsFile)
paths = pickle.load(pathsFile)
featuresFile.close()
labelsFile.close()
pathsFile.close()


# %%
encodedLabels = to_categorical(labels)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    features, encodedLabels, test_size=0.01, random_state=69)


# %%
x_train.shape

# %%

# %%
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(30, 84)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(LSTM(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# %%

# %%
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model.fit(x=x_train, y=y_train, epochs=2, validation_split=0.2)

# %%
modelEvaluate = model.evaluate(x_test, y_test)

# %%
model.save_weights("my.h5")
modelJSON = model.to_json()
with open('my.json', 'w') as jsonFile:
    jsonFile.write(modelJSON)


# %%
def inputProcessing(path):
    features = skeletonExtraction(path)
    temp = [features]

    return np.asarray(temp)


# %%
features = inputProcessing("D:\isl_projects\datasets\Dry\MVI_5167.MOV")
output = model.predict(features)
label = classList[np.argmax(output)]
print(label)

# %%

# %%
jsonFile = open('my.json', 'r')
loadedModel = jsonFile.read()
loadedModel = keras.models.model_from_json(loadedModel)
loadedModel.load_weights('my.h5')
loadedModel.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

# %%
features = inputProcessing("D:\isl_projects\datasets\Dry\MVI_5167.MOV")
output = loadedModel.predict(features)
label = classList[np.argmax(output)]
print(label)

# %%
