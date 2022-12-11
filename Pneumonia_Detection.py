# -*- coding: utf-8 -*-
"""
#Created on Mon Sep 12 16:00:39 2022

#Data derived from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
https://www.kaggle.com/code/homayoonkhadivi/medical-diagnosis-with-cnn-transfer-learning
#@author: hagar chen
"""
# Importing Various Modules.
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet169
from keras.layers import Flatten, Dense, Dropout
from keras import Model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing, metrics

# Defining operation variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# creating full testing\training and validation paths
mainDir = "C:/Users/roeec/Downloads/Data/chest_xray/chest_xray"
testDir = mainDir + "/test"
trainDir = mainDir + "/train"
valDir = mainDir + '/val'

# View Classes
testClasses = os.listdir(testDir)

# Before the learning we will preprocess the images by using ImageDataGenerator object.
# This allows us to do augmentation to avoid over-fitting

Image_gen = ImageDataGenerator(
    rescale=1 / 255,
    shear_range=10,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 2.0],
    width_shift_range=0.2,
    rotation_range=20,
    fill_mode='nearest'
)
Image_gen_test = ImageDataGenerator(rescale=1 / 255)

train = Image_gen.flow_from_directory(trainDir,
                                      batch_size=64,
                                      class_mode='binary',  # classes to predict
                                      target_size=(224, 224),  # resize to this size
                                      shuffle=True
                                      )
test = Image_gen_test.flow_from_directory(testDir,
                                          batch_size=1,
                                          class_mode='binary',
                                          target_size=(224, 224),
                                          shuffle=False
                                          )
val = Image_gen.flow_from_directory(valDir,
                                    batch_size=1,
                                    class_mode='binary',
                                    target_size=(224, 224),
                                    shuffle=False
                                    )
# View a random images
allTrainFiles = train.filepaths
selectedImages = np.random.randint(len(allTrainFiles), size=9)


# createlabels - this function convert the label datatype from logical to categorical.

def createlabels(selectedimages, data):
    logicallabels = data.labels[selectedimages]
    categoricallabels = pd.cut(logicallabels, bins=[-1, 0, 1],
                               labels=['normal', 'pneumonia'])
    return categoricallabels

# fig = plt.figure(figsize=(3, 3))
categoricalLabels = createlabels(selectedImages, train)

# for i in range(0, len(selectedImage)):
#     Image = cv2.imread(allTrainFiles[selectedImage[i]])
#     fig.add_subplot(3, 3, i + 1)
#     # showing images
#     plt.imshow(Image)
#     plt.title(categoricalLabels[i])

# Feature Extraction using DenseNet
featureModel = DenseNet169(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(224, 224, 3),
                           pooling=None,
                           # classes=2,
                           # classifier_activation='softmax'
                           )
for layer in featureModel.layers:
    layer.trainable = False

# Add the final classifier
output = featureModel.output
x = Flatten()(output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', name='dense_1')(x)
x = Dropout(0.5)(x)
# Final Layer (Output)
prediction = Dense(1, activation='sigmoid')(x)

# Creating model object
featureModel = Model(inputs=featureModel.input, outputs=prediction)
featureModel.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy'])
featureModel.summary()
# Training the model
featureModel.fit(x=train, validation_data=val, epochs=3)

# Feature Extraction from the model
# We are taking features from the dropout layer, and use those features to train svm classifier
featureModel = Model(inputs=featureModel.inputs,
                     outputs=featureModel.get_layer(name="dense_1").output)
# Extract training features from the model
x_for_training = featureModel.predict(train)
print(x_for_training.shape)

# Send test data through same feature extractor process
x_test_features = featureModel.predict(test)

# Import and Apply PCA
# Notice the code below has .95 for the number of components parameter.
# It means that scikit-learn choose the minimum number of principal
# components such that 95% of the variance is retained.
pca = PCA(.95)
pca.fit(x_for_training)
x_for_training_pca = pca.transform(x_for_training)
x_test_features_pca = pca.transform(x_test_features)

# Normalize training and testing dataset
scaler = preprocessing.StandardScaler()
scaler.fit(x_for_training_pca)
x_for_training_scaled = scaler.transform(x_for_training_pca)
x_test_features_scaled = scaler.transform(x_test_features_pca)

# checkParameters = 1 if you want to check what are the best values of gamma & c
checkParameters = 0

# check for the best values of C and Gamma
if checkParameters:
    AccAll = []
    CAll = []
    gammaAll = []
    Cval = np.arange(0.1, 3, 0.1)
    gammaVal = np.arange(0.1, 8, 0.1)

    for C in Cval:
        for gamma in gammaVal:
            clf = SVC(kernel='rbf', gamma=gamma, C=C, tol=0.1, class_weight='balanced')
            # Classify the images using svm classifier
            clf.fit(x_for_training_scaled, train.classes)
            # Now predict using the trained svm model
            prediction = clf.predict(x_test_features_scaled)
            Acc = metrics.accuracy_score(test.classes, prediction)
            AccAll.append(Acc)
            gammaAll.append(gamma)
            CAll.append(C)
            metrics.accuracy_score(test.classes, prediction)
else:
    # Classify the images using svm classifier
    C = 0.1
    gamma = 7.9
    clf = SVC(kernel='rbf', gamma=gamma, C=C, tol=0.1, class_weight='balanced')

clf.fit(x_for_training_scaled, train.classes)
prediction = clf.predict(x_test_features_scaled)
metrics.accuracy_score(test.classes, prediction)
print("Accuracy = ", metrics.accuracy_score(test.classes, prediction))
cm = metrics.confusion_matrix(test.classes, prediction)
#sns.heatmap(cm, annot=True)