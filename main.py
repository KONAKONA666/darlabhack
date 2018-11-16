import numpy as np
import pandas as pd
import dlib
import os
from PIL import Image
import pickle


files = [(os.path.join('data', os.path.join(name, f)), name) for name in os.listdir('data') for f in os.listdir(os.path.join('data', name))]
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

es = []

for f in files:

    image = np.asarray(Image.open(f[0]))
    face_detects = face_detector(image, 1)
    if not face_detects:
        print(f)
        continue
    face = face_detects[0]
    landmarks = shape_predictor(image, face)
    embedding = np.asarray(face_recognition_model.compute_face_descriptor(image, landmarks))
    es.append((f[1], embedding))

with open('embeddings.pickle', 'wb') as f:
    pickle.dump(es, f)





