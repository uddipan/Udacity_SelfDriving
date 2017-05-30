# **Finding Lane Lines on the Road** 

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

1. Description of Pipeline:

The lane finding pipeline consisted of the following steps:
 a. GrayScaling
 b. Smoothing
 c. Edge Detection
 d. Region of Interest Selection
 e. Finding Lines
For the final part, the line segments obtained by Hough transform within the ROI 
are considered. The line segments are grouped into several clusters based on their
slopes, e.g. lines with similar slopes (slopes within a constant bound) are grouped
together. The two groups with highest representatives form the left and right lanes.
Thereafter, the line segments belonging to the lanes are averaged and extrapolated
to span the ROI thereby forming a single solid line for each lane.

The results applied to the test images and videos are saved in the folder
test_images_output and test_videos_output respectively.

### 2. Identify potential shortcomings with your current pipeline

One obvious shortcoming is when the road is bending, the slopes of the
line segments change rapidly thereby making the clear distinction between
the two lanes extremely difficult.
Erroneous or almost absent lane markings, painted roads etc.
also cause potential problems.


### 3. Suggest possible improvements to your pipeline

One possible improvement for bending roads would be to identify sections on the lane
and try and stich them to form a continuous line rather than extrapolating. Curve fitting
may also help in that case.
