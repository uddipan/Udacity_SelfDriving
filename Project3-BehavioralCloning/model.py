
# coding: utf-8

# In[1]:


import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:



# Read csv file
# sample
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    first_line = True
    for line in reader:
        if first_line:
            first_line = False
            continue 
        lines.append(line)
        


# In[3]:


# Rotation
def rotate(image, angle):
    rows,cols,depth = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols,rows), 
                            flags = cv2.INTER_NEAREST,
                            borderMode = cv2.BORDER_REPLICATE)
    return rotated

def shear(image):
    rows,cols,depth = image.shape
    M = np.float32([[1,0.2,0],[0.2,1,0]])
    sheared = cv2.warpAffine(image, M, (cols,rows), 
                            flags = cv2.INTER_LINEAR,
                            borderMode = cv2.BORDER_REPLICATE)
    return sheared


# In[4]:


# Load images
def generate(samples, batch_size = 32):
    images_ = []
    measures_ = []
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        batch_counter = 0
        for batch in range(0, num_samples, batch_size):
            batch_samples = samples[batch: batch + batch_size]

            images_ = []
            measures_ = []
            for line in batch_samples:
                correction = [0.0, 0.2, -0.2];
                for i in range(3):
                    image_path = line[i]
                    filename = image_path.split('/')[-1]
                    if filename.endswith('.jpg'):
                        current_path = 'data/IMG/' + filename
                        image = cv2.imread(current_path)
                        images_.append(image)
                        raw_measure = float(line[3])
                        measure = raw_measure + correction[i]
                        measures_.append(measure)
                        images_.append(cv2.flip(image,1))
                        measures_.append(measure*-1.0)
                        if(abs(raw_measure) > 0.14):
                            images_.append(rotate(image,0.5))
                            images_.append(rotate(image,-0.5))
                            measures_.append(measure)
                            measures_.append(measure)
                        if(abs(raw_measure) > 0.28):
                            images_.append(rotate(image,1.5))
                            images_.append(rotate(image,-1.5))
                            measures_.append(measure)
                            measures_.append(measure)
                        
                    
            X_train = np.array(images_)
            y_train = np.array(measures_)
            yield sklearn.utils.shuffle(X_train, y_train) 


# In[5]:


# Display sample image

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

see_test = []
measure_test = []
for i in range(4):
    see_test, measure_test = (next(generate(lines)))

plt.figure(figsize=(5,5))
plt.imshow(np.array(see_test[10]), cmap = "gray")


# In[ ]:


# Build model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer='adam')
train_samples, validation_samples = train_test_split(lines, test_size = 0.05)
batch_size = 32
num_batches_train = batch_size*(int(len(train_samples)/batch_size))
num_batches_validation = batch_size*(int(len(validation_samples)/batch_size))

history_object = model.fit_generator(
    generate(train_samples),
    samples_per_epoch = 19200,
    nb_epoch=5, 
    verbose=1,
    validation_data = generate(validation_samples),
    nb_val_samples = 256
    )

model.save('model_07_21_4_52.h5')
print("Model Saved")


# In[ ]:


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:




