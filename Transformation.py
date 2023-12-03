'''
Face recognition by landmark68.
Then divide the picture into four parts: face, eyebrows, eyes and nose according to the obtained skeleton information.
The original image and the segmented image together serve as the features of the sample.
Store the features and labels of samples in dictionary format.
'''

import cv2
import os
import dlib
import numpy as np
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
import sys
import pickle


# cut out the image
def cut_out(image, landmark):

    x0, x1, y0, y1 = boxing(landmark)

    if x1 > image.shape[1]:
        x1 = image.shape[1] - 1

    if y1 > image.shape[0]:
        y1 = image.shape[0] - 1

    if x0 < 0:
        x0 = 0

    if y0 < 0:
        y0 = 0

    if (abs(x0 - x1) > 224) or (abs(y0 - y1) > 224):

        img = img_resize(image[y0:y1, x0:x1])

    else:

        img = image[y0:y1, x0:x1]

    return img


# chage the shape of image
def img_resize(image):

    height, width = image.shape[0], image.shape[1]
    height_new, width_new = 224, 224

    if width / height >= width_new / height_new:

        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))

    if width / height < width_new / height_new:

        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    return img_new


# Define box for cut out
def boxing(shape):

    x0, y0 = np.min(shape, axis=0)
    x1, y1 = np.max(shape, axis=0)

    return x0, x1, y0, y1

# make image to alignment
def padding(ary):

    z = np.zeros((224, 224))

    x0 = int((224 - ary.shape[0]) / 2)
    x1 = x0 + ary.shape[0]
    y0 = int((224 - ary.shape[1]) / 2)
    y1 = y0 + ary.shape[1]

    z[x0:x1, y0:y1] = ary

    return z


# initialize dlib's face detector (HOG-based) and then create
detector = dlib.get_frontal_face_detector()

# count
count = 0

# the facial landmark predictor
predictor = dlib.shape_predictor('E:\\projekt_1\\shape_predictor_68_face_landmarks.dat')

FACIAL_LANDMARKS_IDXS = face_utils.FACIAL_LANDMARKS_68_IDXS

# Define the path of files
folder_path = "F:/final_dataset/"
folder_names = os.listdir("F:/final_dataset/")


np.set_printoptions(threshold=sys.maxsize)

for name in folder_names:

    file_path = folder_path + name
    file_names = os.listdir(file_path)

    for image_name in file_names:

        image_path = file_path + "/" + image_name

        # load the input image
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # count
        count = count + 1

        # detect faces in the grayscale image
        rects = detector(img_gray, 0)

        # find the exception
        if len(rects) == 0:

            print(image_name)

        else:

            # loop over the face detections
            for (i, rect) in enumerate(rects):

                shape = predictor(img_gray, rect)
                shape = face_utils.shape_to_np(shape)

            # cut out each region as input to the neural network
            try:
                face = cut_out(img_gray, shape[:])
                eyebrow = cut_out(img_gray, shape[17:27])
                eye = cut_out(img_gray, shape[36:48])
                nose = cut_out(img_gray, shape[27:36])
                mouth = cut_out(img_gray, shape[48:])

                # convert image to array

                face_ary = padding(np.array(face, dtype=int))
                eyebrow_ary = padding(np.array(eyebrow, dtype=int))
                eye_ary = padding(np.array(eye, dtype=int))
                nose_ary = padding(np.array(nose, dtype=int))
                mouth_ary = padding(np.array(mouth, dtype=int))

                # save the dataset as dictionary in .txt

                key = ['name', 'face', 'eyebrow', 'eye', 'nose', 'mouth', 'label']
                value = [count, face_ary, eyebrow_ary, eye_ary,
                         nose_ary, mouth_ary, name]
                dic = dict(zip(key, value))

                write_file = open('D:/EMo_DATA_ndarray/'+name+'/'+str(dic['name'])+'.txt', 'wb')

                pickle.dump(dic, write_file, -1)

                write_file.close()
            except Exception as e:
                print(image_name)



