
# coding: utf-8

# In[1]:


# Necessary imports
import numpy as np
import cv2
import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from skimage.feature import hog

from scipy.ndimage.measurements import label


# # Show two images side by side

# In[3]:


def ShowTwoImages(image1, image2, title1, title2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(title1, fontsize=15)
    ax2.imshow(image2)
    ax2.set_title(title2, fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# # HOG Features Extraction and Visualization

# In[2]:


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


# In[ ]:




