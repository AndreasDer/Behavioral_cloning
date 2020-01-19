import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
from random import shuffle

path = "./Recordings"

def process_image(path,filepath):
    #For files recorded on windows machine, \\ instead of / has to be used
    filename = filepath.split('\\')[-1]
    current_path = os.path.join(path,'IMG',filename)
    #print(current_path)
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    
def import_files(path):
    # Correction factor for steering angle measurements
    correction = 0.3 # this is a parameter to tune
    lines = []
    print("Reading file: ",os.path.join(path,'driving_log.csv'))
    with open(os.path.join(path,'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    print(len(lines)," datasets imported.") 
    images = []
    measurements = []
    for line in lines:
        img_center = process_image(path,line[0])
        img_left = process_image(path,line[1])
        img_right = process_image(path,line[2])
        measurement = float(line[3])
        measurement_left = measurement + correction
        measurement_right = measurement - correction
        image_flipped = np.fliplr(img_center)
        measurement_flipped = -measurement
        images.extend((img_center, img_left, img_right,image_flipped))
        measurements.extend((measurement,measurement_left,measurement_right,measurement_flipped))
        #print(measurement," , ",measurement_flipped)
        #print(measurements)
    return images,measurements

def import_samples(path):
    samples = []
    with open(os.path.join(path,'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            samples.append(line)
    print(len(samples)," samples imported.")
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples,validation_samples

def generator(samples, path, batch_size=32):
    # Correction factor for steering angle measurements
    correction = 0.3 # this is a parameter to tune
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_center = process_image(path,batch_sample[0])
                img_left = process_image(path,batch_sample[1])
                img_right = process_image(path,batch_sample[2])
                measurement = float(batch_sample[3])
                measurement_left = measurement + correction
                measurement_right = measurement - correction
                image_flipped = np.fliplr(img_center)
                measurement_flipped = -measurement
                images.extend((img_center, img_left, img_right,image_flipped))
                angles.extend((measurement,measurement_left,measurement_right,measurement_flipped))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    
def print_hist(y_train,path):
    y_train*=25
    fig = plt.hist(y_train,bins=np.arange(-1.0,1.0,0.1))
    plt.savefig(path)

def print_validation_loss(history,path):
    ## plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(path)
    
train_samples, validation_samples = import_samples(path)
# Set the batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples,path, batch_size=batch_size)
validation_generator = generator(validation_samples,path, batch_size=batch_size)

from keras.models import Sequential,load_model
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout,SpatialDropout2D
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/ 255.0 - 0.5))
model.add(Conv2D(24, kernel_size=(5, 5),strides=(2,2),
                 activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(36, kernel_size=(5, 5),strides=(2,2),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

# Uncomment the following line to load the already trained model
#model = load_model('retrained_model_workingep15.h5')

# Uncomment the following line to print out the model summary
#print(model.summary())

# Uncomment the following two lines to plot the model, this only works outside of GPU mode
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

#model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=2)
history_object =model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator,validation_steps=ceil(len(validation_samples)/batch_size),epochs=10, verbose=2)

model.save('retrained_model_working.h5')

# Uncomment to print the validation and training loss
#print_validation_loss(history_object,"validation_loss_working.png")
