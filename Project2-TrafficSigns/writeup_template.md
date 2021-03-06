#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my code:
https://github.com/uddipan/Udacity_SelfDriving/tree/master/Project2-TrafficSigns

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here are 10 random images from the dataset.

![alt text](saved_images/before.png)

Here is an exploratory visualization of the data set. It is a bar chart showing how the
data is classified into different labels.

![alt text](saved_images/hist.png)


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this reduces the dimension from 3 to 1
and most relevant information is embedded in intensity of images. This also helps in speeding up training time.
Next, I center-normalized each image by subtracting each pixel value of an image from the mean pixel value and
dividing the result by the standard deviation of the image. This helps in making the range of pixel values uniform
and avoids heavily skewing a particular subrange.

Here is an example of 10 random traffic sign images after grayscaling and center-normalizing

![alt text](saved_images/preprocess.png)


I decided to generate additional data because increasing the dataset size increases the training accuracy more often than not. This is logical because data augmentation adds more data to classes having fewer elements (i.e. if data pertaining to a particular label is really low, augmentation helps increasing the size of data for that label). 

To add more data to the the data set, I used the following techniques:
1. Rotation : Applied rotations of 7.5 degrees and -7.5 degrees to each original image and created 2x data. The rationale is that images are not always taken at a straight angle and slight rotations of the camera will boost data prediction.
2. Translation: Applied translations of 3 pixels on each of x and y directions to generate 2x data. The rationale is same as rotation.
3. Shear: Applied a random shear to each image to generate 1x data. Shear simulates tilt/yaw/pitch of the camera taking the image
4. Blur: Applied gaussian blur to each image to eliminate the effect of noise. This generated an addition 1x data.

After the pre processing I ended up with 7x original data. The augmented data set is then shuffled to avoid localization of similar labels. Here is an example of 10 random images taken from the augmented dataset.
 
![alt text](saved_images/augmented.png)


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 Grayscale image   			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Average pooling	| 2x2 stride,  outputs 14x14x6 			|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Average pooling	| 2x2 stride,  outputs 5x5x16 (*)			|
| Convolution 5x5 (skip)| 1x1 stride valid padding, outputs 1x1x400     |
| RELU (skip)           |                                               |
| Flatten (*)           | outputs 400                                   |
| Flatten (skip)        | outputs 400                                   |
| Concatenate (skip + *)| outputs 800                                   |
| Dropout               | outputs 800                                   |
| Fully connected	| inputs 800, outputs 43                        |      					|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an approach similar to Lenet. I used an AdamOptimizer, a batch size of 128 (would have tried 256 if the validation accuracy was low), 20 Epochs and a learning rate of 0.001. For the training model convolution layers, I used a mean of 0, and standard deviation of 0.1.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

- I started off by a simple plug and play of the Lenet architecture discussed in class. I used color images at first and obtained a validation accuracy of about 88%. 
- Next I augmented the dataset with translated and rotated images and used the same architecture to get an accuracy of 90%.
- I tried gray scaling and pre processing the dataset as described above and that increased my accuracy to 94%
- Next I tried dropout in the original Lenet architecture just before the final activation and that increased the accuracy further to 95%
- Finally I decided to try the famous architecture from Sermanet and LeCun paper and that further increased my validation accuracy to almost 97.6% with 20 epochs. The final two architectures are documented in the jupyter notebook.


My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 97.4%
* test set accuracy of 96.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture was a simple Lenet plug and play. It was chosen because it is a fast way of validating the correctness of the code and it gave me a basic idea which parts of the entire design can be altered for better accuracy.

* What were some problems with the initial architecture?

First and foremost, desired level of accuracy was not achieved. 
The data was more complex than Lenet and without pre-processing/augmentation accuracy was poor.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Architecture was adjusted first by replacing the max pool by average pool. That boosted the accuracy quite a bit.
Next I tried dropout before the final activation layer
Finally I settled upon the LeCun and Sermanet architecture with skip layers that gave me a validation accuracy of about 97.6%

* Which parameters were tuned? How were they adjusted and why?

I replaced max pool by average pool
In the final architecture a dropout with keep probability of 0.5 was used and only one final activation layer was used. The dropout proved to be significantly helpful.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution will work well as we are dealing with small image subsets and that helps us better identifying local similarities.
Dropout keep probability of 0.5 helps from overfitting.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text](downloaded_images/image1.jpeg) ![alt text](downloaded_images/image2.jpeg) ![alt text](downloaded_images/image3.jpeg)
![alt text](downloaded_images/image4.jpeg) ![alt text](downloaded_images/image5.jpeg) ![alt text](downloaded_images/image6.jpeg) 
![alt text](downloaded_images/image7.jpeg) ![alt text](downloaded_images/image8.jpeg)

Here are the resized and pre-processed versions :
![alt text](saved_images/tests.png)

The main challenges in classification are noises in the image and angles/distances at which they are taken.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		| 30 km/h   				        | 
| Stop sign     	| Stop Sign 					|
| Caution		| Caution					|
| Keep Right	      	| Keep Right					|
| Priority Road		| Priority Road      				|
| Turn Right Ahead      | Turn Right Ahead                              |
| Bumpy Road            | Bicycles crossing                             |
| Right of way ahead    | Right of way ahead                            |


The model was able to correctly guess all but one of the traffic signs, which gives an accuracy of 87.5%. 
This compares favorably to the accuracy on the test set of 96%.

####3. Describe how certain the model is when predicting on each of the new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

All the images except image 7 (Bumpy Road) were successfully classified to their respective labels with softmax probabilities of 100%.
For the ‘Bumpy Road’ image, the top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .47         		| Bicycle crossing   				| 
| .44     		| Bumpy Road 			                |
| .07			| Slippery Road					|
| .02	      		| Road Work					|
| .01			| Dangerous curve to the right     		|


The following image visualizes the soft max probabilities for all images
![alt text](saved_images/softmax.png)

