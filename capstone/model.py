import cv2
import mediapipe as mp
import numpy as np
import os
import copy
import itertools

classList = ['dry', 'healthy', 'sick','Good Morning']
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
sequenceLength = 30

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

def skeletonExtraction(path):

    left = []
    right = []
    cap = cv2.VideoCapture(path)

    while True:
        success, img = cap.read()

        if(success == False):

       
            cap.release()
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        hLeft = None
        hRight = None

        if(results.multi_hand_landmarks):
            for idx, handLms in enumerate(results.multi_hand_landmarks):
                
                hand = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    hand.append([cx, cy])

                label = results.multi_handedness[idx].classification[0].label
                

                if(label == 'Left'):
                    hLeft = normalizeCoordinates(hand)
                elif(label == 'Right'):
                    hRight = normalizeCoordinates(hand)
        
        if(hLeft != None):
            left.append(hLeft)
        if(hRight != None):
            right.append(hRight)
    
    countLeft = len(left)
    countRight = len(right)
    windowLeft = max(countLeft/sequenceLength, 1)
    windowRight = max(countRight/sequenceLength, 1)

    finalFeatures = []

    if countLeft < sequenceLength or countRight < sequenceLength:
        return []


    for i in range(0, sequenceLength):
        
        finalFeatures.append(left[int(i * windowLeft)] + right[int(i * windowRight)])

    return np.asarray(finalFeatures)

def inputProcessing(path):
    featuress = skeletonExtraction(path)
    temp = [featuress]

    return np.asarray(temp)

def predict(file_path)
	featuress = inputProcessing(file_path)
	outpus = loadedModel.predict(featuress)
	labels = classList[np.argmax(outputs)]
	print(labels)