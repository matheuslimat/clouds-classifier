import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fixando as sementes para reproducibilidade
seed = 100
np.random.seed(seed)
tf.random.set_seed(seed)

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
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

input_shape = (img_size[0], img_size[1], 3)
num_classes = 2
model = create_model(input_shape, num_classes)

model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

model.save('cloud_detection_model_v5_with_plot_and_layers.h5')

def plot_history(history):
    # Gráfico de perda
    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Gráfico de acurácia
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

y_pred = model.predict(X_test).argmax(axis=1)
plot_confusion_matrix(y_test, y_pred)

def visualize_layers(model, image):
    """Função para visualizar a saída das camadas de convolução e pooling"""
    
    # Criamos um modelo que retorna a saída de cada camada de convolução, pooling e flatten
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D, Flatten))]
    layer_names = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D, Flatten))]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Executamos o modelo para obter a saída das camadas
    activations = activation_model.predict(image.reshape(1, *image.shape))
    
    for layer_name, layer_activation in zip(layer_names, activations):
        print(f"Output of layer {layer_name}")
        if isinstance(layer_activation, np.ndarray) and len(layer_activation.shape) == 4:
            n_filters = layer_activation.shape[-1]
            print(n_filters)
            n_cols = min(n_filters, 8)
            n_rows = int(np.ceil(n_filters / n_cols))
            fig = plt.figure(figsize=(n_cols * 1.5, 1.5))
            fig.suptitle(layer_name, fontsize=16)
            for i in range(n_filters):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
            plt.show()
        elif isinstance(layer_activation, np.ndarray) and len(layer_activation.shape) == 2:
            n_cols = layer_activation.shape[-1]
            fig2 = plt.figure(figsize=(n_cols * 1.5, 1.5))
            fig2.suptitle(layer_name, fontsize=16)
            ax = plt.subplot(1, 1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(layer_activation[0, :].reshape(1, -1), cmap='viridis', aspect='auto')
            plt.show()

# Escolha uma imagem de teste
image = X_test[0]

# Visualiza as ativações das camadas
visualize_layers(model, image)
