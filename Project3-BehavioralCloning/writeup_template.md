#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 for the final video

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

My model is the nvidia model introduced in class used as is.

The model was trained and validated on different collected data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually 

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road . 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with the provided sample data and Lenet architecture to validate that the car can navigate straight roads. Next I figured that the sample data was not every distributed, and hence collected more turning angles data. 

I used pre-processing (center normalization) and cropping to further fine tune my data. The car was doing much better. 

Then I shifted to nvidia architecture and that improved my performance quite a bit. 

Finally, I did a further pre processing to augment my dataset with images having steering angles greater than a threshold.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is the nvidia architecture.  	
Here are the details for each layer of the model used:
I started with a normalization and cropping layer for batch normalization.
Then I followed nvidi's architecture of a bunch of convolution layers (details below)
and then followed by dense layers introducing a dropout layer as shown below:

![alt text](arch.png)

####3. Creation of the Training Set & Training Process
I mainly collected data from the primary track especially at turning angles.

After the collection process, I had 10K number of data points. I then preprocessed this data by adding more turning angle data, finally having around 60K data points.

Here are some example images of my dataset

![alt text](center_2016_12_01_13_31_13_177.jpg)
![alt text](left_2016_12_01_13_39_24_891.jpg)
![alt text](right_2016_12_01_13_40_07_233.jpg)



Here are some examples of data that I augmented at sharp turnings with slight rotation added
![alt text](t1.jpg)
![alt text](t2.jpg)
![alt text](t3.jpg)

The main motivation was that at turning angles data was sparse and the distribution of turning 
angles were not even. So adding more data at sharp angles helps in training. I added a slight rotation
to each image at sharp turning angles, to augment the data. This helps simulate the turning better
and the intuition can be related to hilly conditions.

I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss curves. Both the training and validation losses were consistent and decreased in 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text](valtr.png)
