import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import cv2

def load_dataset(data_dir, img_size):
    X, y = [], []
    for label, category in enumerate(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        for img_path in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, img_path))
            img_resized = cv2.resize(img, img_size)
            X.append(img_resized)
            y.append(label)
    return np.array(X), np.array(y)

data_dir = 'C://DEV//clouds-classifier//ia//src//data'
img_size = (128, 128)
X, y = load_dataset(data_dir, img_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25)) # adicionando dropout
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25)) # adicionando dropout
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25)) # adicionando dropout
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) # aumentando o dropout
    model.add(Dense(num_classes, activation='softmax'))
    return model

input_shape = (img_size[0], img_size[1], 3)
num_classes = 2
model = create_model(input_shape, num_classes)

model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=300, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

model.save('cloud_detection_model.h5')

