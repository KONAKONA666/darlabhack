import numpy as np
import pickle
import operator
import os

from PIL import Image
import dlib

ds = []
curr_img = np.asarray(Image.open('test.jpg'))

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

face_detects = face_detector(curr_img, 1)
face = face_detects[0]
landmarks = shape_predictor(curr_img, face)
curr = np.asarray(face_recognition_model.compute_face_descriptor(curr_img, landmarks))


with open('embeddings.pickle', 'rb') as f:
    embeddings = pickle.load(f)
    for name, embedding in embeddings:
        distance = np.linalg.norm(curr - embedding)
        ds.append((distance, name))

    best_match = min(ds)
    print(best_match)
