
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


# Rotation: Adds a random 
def rotate(image, angle):
    rows,cols,depth = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols,rows), 
                            flags = cv2.INTER_NEAREST,
                            borderMode = cv2.BORDER_REPLICATE)
    return rotated


# In[4]:


# Load images
# Sample generator to load image files
def generate(samples, batch_size = 32):
    images_ = []
    measures_ = []
    num_samples = len(samples)

    while 1:
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
                    # only deal with images ending in jpg
                    if filename.endswith('.jpg'):
                        current_path = 'data/IMG/' + filename
                        image = cv2.imread(current_path)
                        images_.append(image)
                        raw_measure = float(line[3])
                        measure = raw_measure + correction[i]
                        measures_.append(measure)
                        images_.append(cv2.flip(image,1))
                        measures_.append(measure*-1.0)
                        # add more images if greater than threshold
                        # as distribution is not even
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


# In[12]:


# Build model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

# This is the nvidia architecture with minor variation

model = Sequential()

# Add batch normalizations to avoid over fitting
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Add cropping to crop off top and bottom portions of each image
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Add convolutions
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

# Add final flatten and dense layers
model.add(Flatten())
model.add(Dense(100))

# Add dropout 
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Print a summary of the model
model.summary()

model.compile(loss='mse',optimizer='adam')
train_samples, validation_samples = train_test_split(lines, test_size = 0.05)
batch_size = 32
num_batches_train = batch_size*(int(len(train_samples)/batch_size))
num_batches_validation = batch_size*(int(len(validation_samples)/batch_size))

# Fit the model by invoking the generator
history_object = model.fit_generator(
    generate(train_samples),
    samples_per_epoch = 19200,
    nb_epoch=5, 
    verbose=1,
    validation_data = generate(validation_samples),
    nb_val_samples = 256
    )

# Save Model
model.save('model_07_21_7_04.h5')
print("Model Saved")


# In[27]:


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:




