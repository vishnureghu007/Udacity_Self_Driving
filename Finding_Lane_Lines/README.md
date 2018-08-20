# **Finding Lane Lines on the Road** 
In this project you will detect lane lines in images using Python and OpenCV.
---
The goals / steps of this project are the following:
---
1)Writing the image processing functions using opencv for detecting lane from the input road image.
---
2)Testing with set of images 
---
3)Testing for video frames 
---
4)Writing the output annotated video
---

[//]: # (Image References)

[image1]: ./intermediate/gray.png "Grayscale"
[image2]: ./intermediate/blur_gray.png "blur"
[image3]: ./intermediate/edges.png "Canny Edge"
[image4]: ./intermediate/final_output.png "hough line"



---
# Pipeline

Consisted of below steps.
---
1-Color image to gray
![alt text][image1]
---
2-gaussian_blur
![alt text][image2]
---
3-Canny edge detection
![alt text][image3]
---
4-Hough tranform for detecting lines(Removing slope outliers and extrpolate)
![alt text][image4]
---


# Possible Improvements

The above code tested for only straight roads. Improvements are possible for more complex scenarios and curvy roads.
---
https://airccj.org/CSCP/vol5/csit53211.pdf
References:
---
https://github.com/udacity/CarND-LaneLines-P1
---
