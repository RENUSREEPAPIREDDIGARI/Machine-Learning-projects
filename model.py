import os #working with file system
import sklearn #for ML utilities
import cv2 #openCV for image processing
import matplotlib.pyplot as plt #for plotting images
import tensorflow as tf #fr DL

data_dir = "Alzheimer_s Dataset" #set the data dir to print contents
print(os.listdir(data_dir))

data_dir += "/train"
print(os.listdir(data_dir))

temp_dir = data_dir + "/MildDemented" #load and display image


for img in os.listdir(temp_dir):
    img_array = cv2.imread(os.path.join(temp_dir, img))
    #print(img_array)
    plt.imshow(img_array)
    plt.show()
    break

print(img_array.shape)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.8, 1.2], zoom_range=[0.99, 1.01], horizontal_flip=True, fill_mode="constant", data_format="channels_last") #data aug


train_data_gen = image_generator.flow_from_directory(directory=data_dir, target_size=(176, 176), batch_size=6500, shuffle=False)

import numpy as np
from random import randint

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def show_images(generator,y_pred=None):
    #Input: An image generator, predicted labels (optional)
    #Output: Displays a grid of 9 random images with lables
        
    #get image lables
    labels =dict(zip([0,1,2,3], classes))
    
    #get a batch of images
    x,y = generator.next()
    
    #display a grid of 9 images
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            idx = randint(0, 5121) #because there are a total of 5121 images
            plt.imshow(x[idx].astype('uint8')) #This is done because of the reason that if the color intensity is a float,
            #then matplotlib expects it to range from 0 to 1. If an int, then it expects 0 to 255.
            #So we can either force all the numbers to int or scale them all by 1/255 or use the .astype function to cast the object onto our specified dtype.
            plt.axis("off")
            plt.title("Class: {}".format(labels[np.argmax(y[idx])]))
                                                     
    else:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i].astype('uint8'))
            plt.axis("off")
            plt.title("Actual: {} \nPredicted: {}".format(labels[np.argmax(y[i])],labels[y_pred[i]]))
    
show_images(train_data_gen)


train_data, train_labels = train_data_gen.next()
print(train_data.shape, train_labels.shape)


for alzheimers_class in classes:
    container = []
    temp_dir = data_dir + "/" + alzheimers_class
    for file in os.listdir(temp_dir):
        container.append(file)
    print(alzheimers_class, ": ", len(container))


import imblearn

sm = imblearn.over_sampling.SMOTE(random_state=42) #synthetic minority over sampling technique to handle class imbalance
train_data, train_labels = sm.fit_resample(train_data.reshape(-1, 176 * 176 * 3), train_labels)
train_data = train_data.reshape(-1, 176, 176, 3)
print(train_data.shape, train_labels.shape)

def conv_block(filters, act='relu'):    #defining the arch
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(filters, 3, activation=act, padding='same'))
    block.add(tf.keras.layers.Conv2D(filters, 3, activation=act, padding='same'))
    block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.MaxPool2D())
    
    return block

def norm_block(units, dropout_rate, act='relu'):    
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Dense(units, activation=act))
    block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.Dropout(dropout_rate))
    
    return block


def construct_model(act='relu'):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*[176, 176], 3)),
        tf.keras.layers.Conv2D(16, 3, activation=act, padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation=act, padding='same'),
        tf.keras.layers.MaxPool2D(),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        norm_block(512, 0.7),
        norm_block(128, 0.5),
        norm_block(64, 0.3),
        tf.keras.layers.Dense(4, activation='softmax') #Output Layer       
    ], name = "cnn_model")

    return model

model = construct_model()

METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc')]
    
model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)

model.summary()

history = model.fit(train_data, train_labels, epochs=100, validation_split=0.2)

fig, ax = plt.subplots(1, 3, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['auc', 'loss', 'acc']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])


data_dir1 = data_dir[:19]+'/test'
data_dir1

test_data_gen = image_generator.flow_from_directory(directory=data_dir1, target_size=(176, 176), batch_size=6500, shuffle=False)

test_data, test_labels = test_data_gen.next()
print(test_data.shape, test_labels.shape)


test_scores = model.evaluate(test_data, test_labels)
test_scores

predicted_test_labels = model.predict(test_data)

def roundoff(arr):
    """To round off according to the argmax of each predicted label array. """
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in predicted_test_labels:
    labels = roundoff(labels)

print(sklearn.metrics.classification_report(test_labels, predicted_test_labels, target_names=classes))

pred_ls = np.argmax(predicted_test_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = sklearn.metrics.confusion_matrix(test_ls, pred_ls)
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

import seaborn as sns
ax = sns.heatmap(conf_arr, cmap='Reds', annot=True, fmt='d', xticklabels=classes, yticklabels=classes)

plt.title('Alzheimer\'s Disease Diagnosis')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show(ax)

model.save("model")







