'''
Batch read data.
'''

import os
import random
import numpy as np
import pickle
import tensorflow as tf


# traversing files to generate file paths
folder_path = "EMo_DATA_ndarray/"
file_names = os.listdir(folder_path)
random.seed(5)
random.shuffle(file_names)
label_code = {'Angry': 0, 'Disgust': 1,
              'Fear': 2, 'Happy': 3,
              'Neutral': 4, 'Sad': 5,
              'Surprise': 6}

# generate sample and label lists
def read_datasets(batch_size, type, num_input):
    # Leave 10% of the data for the validation set
    if type == "train_datas":
        start = 0
        end = -3500
    elif type == "val_datas":
        start = -3500
        end = 35000
    while 1:
        face_x = []
        eyebrow_x = []
        eye_x = []
        nose_x = []
        mouth_x = []
        label_y = []
        count = 0

        # Loop to fetch data
        for file_name in file_names[start:end]:

            file_path = folder_path + file_name
            fr = open(file_path, 'rb')
            data = pickle.load(fr)
            fr.close()

            if num_input == 5:
                face_x.append(data['face'])
                eyebrow_x.append(data['eyebrow'])
                eye_x.append(data['eye'])
                nose_x.append(data['nose'])
                mouth_x.append(data['mouth'])
                label_y.append(label_code[data['label']])
                count += 1

                if count == batch_size:
                    count = 0
                    yield ({'input_1': np.array(face_x), 'input_2': np.array(eyebrow_x),
                            'input_3': np.array(eye_x), 'input_4': np.array(nose_x),
                            'input_5': np.array(mouth_x)}, tf.one_hot(label_y, 7))
                    face_x = []
                    eyebrow_x = []
                    eye_x = []
                    nose_x = []
                    mouth_x = []
                    label_y = []

            if num_input == 4:
                eyebrow_x.append(data['eyebrow'])
                eye_x.append(data['eye'])
                nose_x.append(data['nose'])
                mouth_x.append(data['mouth'])
                label_y.append(label_code[data['label']])
                count += 1

                if count == batch_size:
                    count = 0
                    yield ({'input_1': np.array(eyebrow_x), 'input_2': np.array(eye_x),
                            'input_3': np.array(nose_x), 'input_4': np.array(mouth_x)}, tf.one_hot(label_y, 7))
                    eyebrow_x = []
                    eye_x = []
                    nose_x = []
                    mouth_x = []
                    label_y = []

            if num_input == 1:
                face_x.append(data['face'])
                label_y.append(label_code[data['label']])
                count += 1

                if count == batch_size:
                    count = 0
                    yield ({'input_1': np.array(face_x)}, tf.one_hot(label_y, 7))
                    face_x = []
                    label_y = []

