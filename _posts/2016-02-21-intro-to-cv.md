---
layout: default
title:  "Intro to CV"
date:   2016-02-21 13:50:00
categories: main
---

<head>
<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
</head>

# Intro to Computer Vision 

This article covers the basics of 2D computer vision, including image representation, filters, edge detection, feature detectors, and segmentation. 
Later posts will cover deep learning topics.
It is heavily based on slides from Professor Fei-Fei Li's class, CS131. Its purpose is mostly to be notes and review for myself, but hopefully it might be useful for other people as well! 

## Table of Contents
* [Image Representation](#image-representation)
* [Image Filters](#image-filters)
* [Convolution](#convolution)
* [Edge Detection](#edge-detection)
	- [Canny Edge Detector](#canny-edge-detector)
	- [RANSAC](#ransac)
* [Feature Descriptors](#feature-descriptors)
	- [Global vs. Local Feature Descriptors](#global-vs-local-feature-descriptors)
	- [Global Feature Descriptors](#-global-feature-descriptors)
    * [Histogram of Oriented Gradients (HOG)](#histogram-of-oriented-gradients-hog)
	- [Local Invariant Features](#local-invariant-features)
		* [Harris Corner Detector](#harris-corner-detector) 
		* [Laplacian of Gaussian (LoG)](#laplacian-of-gaussian-log)
		* [Difference of Gaussians (DoG)](#difference-of-gaussians-dog)
		* [SIFT](#sift)


# Image Representation

The **goal** of computer vision is to bridge the gap between pixels and "meaning." When you give a computer an image, all it sees is a 2D (or 3D, if the image is in color) numerical array:

<img src="http://images.slideplayer.com/16/5003478/slides/slide_7.jpg" style="width:500px;"/>

## What kind of information can we extract from an image? 

### Metric Information

* 3D modeling
* Structure from motion
* Shape and motion capture
* etc.

<img src="http://www.cs.cornell.edu/projects/disambig/img/disambig_cover.png" style="width:700px;"/>

### Semantic Information

* Image classification
* Object detection
* Semantic image segmentation
* Activity classification
* Facial recognition
* Optical character recognition (OCR)
* etc.

<img src="https://i.ytimg.com/vi/WZmSMkK9VuA/hqdefault.jpg" style="width:500px;"/>

## How do we represent an image? 
An image contains a discrete number of pixels, each of which have an "intensity" value: 
* Grayscale: [0, 255]
* Color: [R, G, B] 

The image is represented as a matrix of integer pixel values.

<img src="http://openframeworks.cc/ofBook/images/image_processing_computer_vision/images/lincoln_pixel_values.png" style="width:800px"/>

(I bet you realized this was a picture of Abraham Lincoln. The human brain is amazing!)

### What information is "useful" in an image?
* Edges
* Corners
* Color information
* etc. (to be finished)

# Image Filters

A (linear) image filter is used to form a new image whose pixels are a weighted combination of the image's original pixel values. The **goals** of applying filters are to:
* Extract useful **features** from the original image (e.g., edges, corners)

<div style="width:100%">
  <div style="margin: 0 auto; width:50%">
    <img src="http://www.bogotobogo.com/python/OpenCV_Python/images/Canny/Canny_Edge_Detection.png" style="height:200px"/>
  </div>
</div>


* **Modify** or enhance the original image (e.g., de-noising, super-resolution)

<div style="display:inline-block;">
	<div style="float:left">
		<img src="http://znah.net/images/TV_denoise/TV_denoise_fig_00.png" style="height:200px"/>
		<p style="width:350px;text-align:center;font-size:14px">De-noising</p>
	</div>
	<div style="float:left;margin-left:20px">
		<img src="http://electronicimaging.spiedigitallibrary.org/data/journals/electim/927109/jei_22_4_041120_f004.png" style="height:200px"/>
		<p style="width:350px;text-align:center;font-size:14px">Super-resolution</p>
	</div><br>
</div>

## Filter example #1: Moving Average

This filter replaces each pixel with an average of its neighboring pixels. The affect is to smooth the image (remove sharp features). 

Below is how we compute the moving average over a 3x3 window:

$$
\begin{split}
g[n, m] &= \frac{1}{9} \sum_{k=n-1}^{n+1} \sum_{l=m-1}^{m+1} f[k, l] \\ 
&= \frac{1}{9} \sum_{k=-1}^{1} \sum_{l=-1}^{1} f[n-k, m-l]
\end{split}
$$

<div style="width:100%">
  <div style="margin: 0 auto; width:80%">
    <img src="https://i.stack.imgur.com/PnWe2.png" style="width:1000px"/>
  </div>
</div>

## Filter example #2: Image Segmentation

Image segmentation based on a pixel intensity threshold:

$$
g[n, m] 
\begin{cases}
255 & \text{ if } f[n, m] > 100\\
0 & \text{ otherwise }
\end{cases}
$$

Image after segmentation:

<img src="http://vgl-ait.org/cvwiki/lib/exe/fetch.php?media=opencv:tutorial:simple.jpg" style="height:250px"/>/

The thresholding filter is not actually linear, because we cannot recreate the new image, $$G$$, from the old image $$F$$ with a fixed set of weights. 

# Convolution

Convolution is the process of adding each element of an image to its local neighbors, weighted by a [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution) (essentially, a small matrix used to apply effects to an image, such as sharpening, blurring, or outlining). It is **not** traditional matrix multiplication. 

[This article](http://setosa.io/ev/image-kernels/) has great visuals to help explain image kernels.
## Example convolutions

$$\text{Image}
* \begin{array}{|c|c|c|}
\hline
0 & 0 & 0 \\ \hline
0 & 1 & 0 \\ \hline
0 & 0 & 0 \\ \hline
\end{array}
= \text{Unchanged image}
$$

$$\text{Image}
* \begin{array}{|c|c|c|}
\hline
0 & 0 & 0 \\ \hline
0 & 0 & 1 \\ \hline
0 & 0 & 0 \\ \hline
\end{array}
= \text{Shifted right by 1 pixel}
$$

$$\text{Image}
* 
\frac{1}{9}
\begin{array}{|c|c|c|}
\hline
1 & 1 & 1 \\ \hline
1 & 1 & 1 \\ \hline
1 & 1 & 1 \\ \hline
\end{array}
= \text{Blurred image}
$$

$$\text{Image}
* 
\frac{1}{9}
\begin{array}{|c|c|c|}
\hline
0 & 0 & 0 \\ \hline
0 & 2 & 0 \\ \hline
0 & 0 & 0 \\ \hline
\end{array}
- \frac{1}{9}
\begin{array}{|c|c|c|}
\hline
1 & 1 & 1 \\ \hline
1 & 1 & 1 \\ \hline
1 & 1 & 1 \\ \hline
\end{array}
= \text{Sharpened image}
$$

# Edge Detection

We know that edges are special from vision studies ([Hubel & Wiesel, 1960s](http://hubel.med.harvard.edu/papers/HubelWiesel1964NaunynSchmiedebergsArchExpPatholPharmakol.pdf)). Edges encode most of the semantic and shape information of an image.  

The **goal** of edge detection is to identify sudden changes (edges) in an image. Ideally, we want to recover something like an artist line drawing.

<img src="http://www.clipartbest.com/cliparts/9iz/o4b/9izo4bRET.png" style="width:200px;"/>

## What causes edges?

<img src="https://image.slidesharecdn.com/finalminorprojectppt-140422115839-phpapp02/95/fuzzy-logic-based-edge-detection-11-638.jpg?cb=1398168182" style="width:1000px;"/>

## How do we characterize edges?

**Definition**: An edge is a place of rapid change in the image intensity function. Edges correspond to the extrema of the first derivative. 

<img src="https://mipav.cit.nih.gov/pubwiki/images/1/11/EdgeDetectionbyDerivative.jpg" style="width:200px"/>

## Image gradients

The gradient of an image is given by:

$$ \nabla f = \Big[ \frac{\partial f}{\partial x}, \frac{\partial y}{\partial x} \Big] $$

where $$x$$ and $$y$$ are the two axes in the 2D image plane. The gradient vector points in the direction of the most rapid increase in intensity. 

<img src="{{ site.baseurl }}/assets/images/image_grads.png"/>

The **gradient direction** is given by:

$$ \theta = \tan^{-1} \Big( \frac{\partial f}{\partial y} \big/ \frac{\partial f}{\partial x} \Big) $$

The **edge strength** is given by:

$$ 
\Vert \nabla f \Vert = \sqrt{\Big( \frac{\partial f}{\partial x} \Big)^2 
+ \Big( \frac{\partial f}{\partial y} \Big)^2} 
$$

Computing and plotting the gradients this way is called **finite differences.**

<img src="{{ site.baseurl }}/assets/images/finite_diffs.png"/>

## Effects of noise

Consider a single row or column of pixels in the image. We can plot the intensity as a function of position:

<img src="{{ site.baseurl }}/assets/images/image_grads_noise.png" style="width:500px"/>

How do we tell where the edge is, given the graph for $$ \frac{d}{dx} f(x) $$? We can't, because finite difference filters respond very strongly to noise.

**Solution**: we can smooth the image, by forcing pixels that look different than their neighbors to look more like their neighbors (remember the blurring convolution filter?) 

To smooth the image, we will use a specific kind of filter, the **Gaussian kernel**. In one dimension the Gaussian kernel is:

$$
G_{\sigma}(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{x^2}{2 \sigma^2}}
$$

where $$\sigma$$ is the standard deviation of the Gaussian distribution. In two dimensions, the Gaussian kernel is the product of two 1D Gaussians, one in each dimension:

$$
G_{\sigma}(x, y) = \frac{1}{2 \pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
$$

<img src="{{ site.baseurl }}/assets/images/gaussian_kernel.png" style="width:500px"/>

Below is the same image with Gaussian kernels applied to it, each with a different value of $$\sigma$$: 

<img src="{{ site.baseurl }}/assets/images/owls.png" style="width:800px"/>

Once we apply the Gaussian kernel $$g$$ to the original image, we get a set of graphs like this: 

<img src="{{ site.baseurl }}/assets/images/smoothed_image_grads.png" style="width:500px"/>

Then, to find edges, we just look for peaks in $$ \frac{d}{dx} (f * g) $$.

## Problems with this simple edge detection 

Smoothing the derivative removes noise, but it also blurs the edge, and causes us to find edges at different "scales," depending on the amount of smoothing. 

<img src="{{ site.baseurl }}/assets/images/noise_to_smooth.png" style="width:500px"/> 

The gradient magnitude is large along a thick "ridge" of pixels, so how do we identify the true edge?

## Canny Edge Detector

This is likely the most widely used edge detector in computer vision. We will go through each step of the algorithm individually. 

### Step 1: Noise reduction

Since edge detection is susceptible to noise in the image, we first remove noise with a Gaussian filter (for some chosen value of $$\sigma$$):

$$
g \gets \text{image} * G_{\sigma}(x, y) 
$$ 

### Step 2: Find magnitude and orientation of the intensity gradient 

For every pixel in the image, we can find the gradient orientation and magnitude as follows:

Gradient orientation:
$$ \theta = \tan^{-1} \Big( \frac{\partial g}{\partial y} \big/ \frac{\partial g}{\partial x} \Big) $$

Gradient magnitude (also known as **edge strength**):
$$
\Vert g \Vert = \sqrt{\Big( \frac{\partial g}{\partial x} \Big)^2 + \Big( \frac{\partial g}{\partial y}\Big)^2{}}
$$

The gradient direction will always be perpendicular to edges.

### Step 3: Non-maximum suppression

After getting the gradient magnitude and direction, we do a full scan of the image to remove any pixels that may not constitute the edge. This allows us to get "thin," single pixel-wide edges rather than the thick ridge of edge pixels we saw before.

For every pixel in the image, we do the following:

1. Compare the gradient magnitude of the current pixel with the gradient magnitude of the pixels in the positive and negative gradient directions.
2. If the gradient magnitude of the current pixel is the largest, then its value will be preserved. Otherwise, the pixel will be suppressed (by setting its gradient value to 0).
 
_
![](http://docs.opencv.org/3.1.0/nms.jpg)

<div style="display:inline-block;">
  <div style="float:left">
    <img src="http://www-scf.usc.edu/~boqinggo/Canny/gradient_lena.jpg" style="height:300px"/>
    <p style="width:300px;text-align:center;font-size:14px">Before non-max suppression</p>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="http://www-scf.usc.edu/~boqinggo/Canny/nonmaximum_suppress_lena.jpg" style="height:300px"/>
    <p style="width:300px;text-align:center;font-size:14px">After non-max suppression</p>
  </div>
</div>

### Step 4: Hysteresis thresholding

The final stage decides which of the edges are really edges and which are not. For this, we need two threshold values, `minVal` and `maxVal`, which must be chosen carefully.

Any edges with intensity gradient magnitude more than `maxVal` are classified as "definite" edges, and those below `minVal` are discarded. The edges that lie between these two thresholds are classified edges or non-edges based on their connectivity. If they are connected to "definite-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded. See the image below: 

<img src="http://docs.opencv.org/trunk/hysteresis.jpg" style="width:400px"/>

The edge $$A$$ is has a gradient magnitude that is greater than `maxVal`, so it is a "definite edge." Edge $$C$$ above the threshold `minVal`, so it is not immediately discarded. $$C$$ is below the threshold `maxVal`, but it is connected to $$A$$ so it is classified as an edge. Edge $$B$$ is above `minVal`, but it is not connected to any "definite edge," so it is discarded. 

After hysteresis, we end up with only the strong edges in the image:

<div style="display:inline-block;">
  <div style="float:left">
    <img src="http://www-scf.usc.edu/~boqinggo/Canny/nonmaximum_suppress_lena.jpg" style="height:300px"/>
    <p style="width:300px;text-align:center;font-size:14px">Before hysteresis</p>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="http://www-scf.usc.edu/~boqinggo/Canny/canny_lena.jpg" style="height:300px"/>
    <p style="width:300px;text-align:center;font-size:14px">After hysteresis</p>
  </div>
</div>

We can do all of these steps in one line in Python:

{% highlight python%}
import cv2
import numpy as np

img = cv2.imread('fimg_name.jpg', 0)  # Loads an image in grayscale
edges = cv2.Canny(img, 100, 200)  # 2nd and 3rd args are minVal and maxVal, respectively
{% endhighlight %}

## RANSAC 

### Line fitting

Sometimes we want to fit lines in an image, rather than just detect edges. Many objects are characterized by the presence of straight lines:

<div style="display:inline-block;">
  <div style="float:left">
    <img src="{{ site.baseurl }}/assets/images/line-fit1.png" style="height:300px"/>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="{{ site.baseurl }}/assets/images/line-fit2.png" style="height:300px"/>
  </div>
</div>

Edge detection doesn't solve this completely, because it poses problems of its own:
* Introduces clutter (which points go with which line, if any?)
* It misses some parts of a line (discontinuities)
* There is noise in the measured edge points and orientations (how do we detect the true parameters, e.g., line slope and y-intercept?) 

<div style="width:100%">
  <div style="margin: 0 auto; width:50%">
    <img src="{{site.basurl}}/assets/images/edge-detection-building.png" style="height:400px"/>
  </div>
	<p style="width:85%;text-align:center;font-size:14px">Where are the straight lines in this image?</p>
</div>

### Voting

It's not feasible to check all combinations of features by fitting a model to each possible subset. **Voting** is a general technique where we let the features "vote" for all models that are compatible with them:
* Loop through each set of features, and for each feature set, cast votes for model parameters
* Choose the model parameters that receive the most votes 

Features that arise from noise or clutter will also get to cast votes, but typically their votes should be inconsistent with the majority of "good" features.

### RANSAC (**RAN**dom **SA**mple **C**onsensus)

**RANSAC** is an iterative method to estimate the parameters of a mathematical model from a set of observed data that contains **inliers** and **outliers**. Inliers are data whose distribution can be explained by a set of model parameters, and outliers are data that do not fit the model (e.g., extreme noise, incorrect measurements). Here, we're trying to estimate the parameters for a line in an image.

The RANSAC algorithm (for the task of line fitting) is as follows:

{% highlight python%}
import sys
import random 
from scipy import stats

def ransac(data, k, n, t, d):
  """
  RANSAC algorithm
  
  Args:
    data: A set of observed data points (list of (x, y) tuples).
    k: The maximum number of iterations allowed in the algorithm. 
    n: The minimum number of data values required to fit the model.
    t: A threshold value for determining when a data point fits a model. 
    d: The number of inliers required to assert that the model fits the data well.

  Returns: 
    This is what is returned.
  """
  best_slope, best_intercept = 0
  best_error = sys.maxint

  for i in range(k):  
    # Step 1: Randomly sample a set of n points
    possible_inliers = random.sample(data, n)
    
    # Step 2: Estimate the least-squares fit line from these randomly sampled points
    x_data, y_data = zip(*possible_inliers)
    slope, intercept, _, _, _ = stats.linregress(x_data, y_data)

    # Step 3: Find the definite inliers for this line, by getting the set of points
    # (not in the random sample) within a certain threshold, t, of the line
    definite_inliers = []
    for point in data:
      if point not in possible_inliers:
        x0 = point[0]
        y0 = point[1]
        dist_from_line = abs(-slope*x0 + y0 + intercept) / sqrt(slope**2 + 1) 
        if dist_from_line < t:
          definite_inliers.append(point)
     
    # Step 4: If the number of inliers is sufficiently large (>= d), then this 
    # line may be a good fit
    if len(inliers) >= d:
      # Refit the line using all points in possible_inliers and definite_inliers
      x_data, y_data = zip(*(possible_inliers + definite_inliers))
      better_slope, better_intercept, _, _, error = stats.linregress(x_data, y_data) 

      # Keep track of the best fit line found so far
      if error < best_error:
        best_slope = better_slope
        best_intercept = better_intercept
        best_error = error 

  return best_slope, best_intercept
{% endhighlight %}

We calculate `dist_from_line` using [this formula](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line):

$$
\text{distance}(ax + by + c = 0, (x_0, y_0)) = \frac{\mid a x_0 + b y_0 + c \mid}{\sqrt{a^2 + b^2}}
$$

$$
\text{distance}(-mx + y -b = 0, (x_0, y_0)) = \frac{\mid -m x_0 + y_0 + b \mid}{\sqrt{m^2 + 1}}
$$

Below are visuals for each of the 4 steps to the RANSAC algorithm:

<div style="display:inline-block;">
  <div style="float:left">
		<p style="text-align:center;font-size:14px">RANSAC Step 1</p>
    <img src="{{ site.baseurl }}/assets/images/ransac-1.png" style="height:200px"/>
  </div>
  <div style="float:left;margin-left:20px">
		<p style="text-align:center;font-size:14px">RANSAC Step 2</p>
    <img src="{{ site.baseurl }}/assets/images/ransac-2.png" style="height:200px"/>
  </div>
	<div style="float:left">
    <p style="text-align:center;font-size:14px">RANSAC Step 3</p>
    <img src="{{ site.baseurl }}/assets/images/ransac-3.png" style="height:180px"/>
  </div>  
	<div style="float:left;margin-left:20px">   
		<p style="text-align:center;font-size:14px">RANSAC Step 4</p>
    <img src="{{ site.baseurl }}/assets/images/ransac-4.png" style="height:180px"/>
  </div>
</div>

### How to choose $$k$$?

The last thing we need to do before running the algorithm is to choose values for the hyperparameters $$k$$, $$n$$, $$t$$, and $$d$$. 
The latter two $$t$$ and $$d$$, can be determined from experimental evaluation, because they tend to be application-specific.
$$n$$ is the minimum number of points needed to fit the model ($$n=2$$ for fitting a line). 

The hyperparameter we really need to optimize is $$k$$, the number of samples needed. 

Let $$p$$ be the probability that, *in some iteration*, the RANSAC algorithm selects **only inliers** from the input data set when it chooses the random sample of $$n$$ points. Therefore, $$p$$ is the probability that, at some point, the algorithm will produce a "good" model.

Let $$w$$ be the probability of choosing an inliner each time a single point is selected, that is,

$$ w = \frac{\text{# inliers in data}}{\text{# points in data}} $$

We may not know $$w$$ a priori, but we can estimate its value. Assuming that the $$n$$ random points are chosen independently, $$w^n$$ is the probability that all $$n$$ points are inliners and $$1-w^n$$ is the probability that at least one of the $$n$$ points is an outlier (and therefore, the probability that a "bad" model will be estimated from `possible_inliers` and `definite_inliers`).

Since we choose $$k$$ samples during our $$k$$ iterations, the probability that the algorithm never selects a set of $$n$$ points which all are inliers is:

$$ 1 - p = (1-w^n)^k$$

So, we can choose $$k$$ to be high enough to keep $$1-p$$ below a desired failure rate. 

### Pros and cons of RANSAC

**Pros**
* RANSAC is a general method suited for a wide range of applications.
* It's easy to implement
* It's easy to calculate its failure rate. 

**Cons**
* RANSAC can only	handle a moderate	percentage of	outliers ([< 50%](https://en.wikipedia.org/wiki/Random_sample_consensus)) without its runtime blowing up.	
* Many real problems have a high rate of outliers. 

# Feature Descriptors 

Features are the information extracted from images in terms of numerical values that are difficult for humans to understand and correlate, but are easy for a computer to "understand." 
In computer vision, feature are sometimes called descriptors. We will use these two terms interchangeably.

Generally, the feature extracted from an image are of a much lower dimension than the original image. This reduction in dimensionality reduces the computation of processing the batch of image. 

Typically, a feature descriptor converts an image of size width x height x 3 (color channels) to a feature vector of length $$n$$. 

We then can feed these feature vectors into any kind of model we want (e.g., Support Vector Machines) for a variety of problem domains! (e.g., object recognition, image classification). 

## Global vs. Local Feature Descriptors
A feature descriptor encodes an image in a way that allows it to be compared and matched to other images.

A **global descriptor** describes the entire image. Global features include contour representations, shape descriptors, and texture features. 
An example of a global feature descriptor is the [Histogram of Oriented Gradients](#histogram-of-oriented-gradients-hog) (HOG). 
Global feature descriptors are generally not very robust, as a change in part of the image may cause it to fail.

A **local descriptor** describes a patch within an image. Multiple local descriptors are used to match an image. This makes them more robust to changes between the matched images (e.g., lighting, deformation, occlusion, etc.). [SIFT](#sift) and SURF are good examples of local features.

Generally, for low-level applications such as object detection (determining whether or not an object exists in an image), image classification, and image retrieval, global features are used. For higher-level applications like object recognition (recognizing the identity of a person/object in an image), local features are used. Combining global and local features improves the accuracy of the recognition, with the side effect of having a heavier computation overhead.

## Existing detectors

There are many existing detectors available, and they have become a basic building block  for
many recent applications in Computer Vision.  
* Corner detection: 
  - [Hessian](https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian) [Beaudet '78]
  - [Harris](https://en.wikipedia.org/wiki/Corner_detection#The_Harris_.26_Stephens_.2F_Plessey_.2F_Shi.E2.80.93Tomasi_corner_detection_algorithms) [Harris '88]
* Blob detectors: 
  - [Laplacian of Gaussian (LoG)](https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian) [Lindeberg '98]
  - [Difference of Gaussians (DoG)](https://en.wikipedia.org/wiki/Difference_of_Gaussians) [Lowe '99]
  - [Hessian-/Harris-Laplacian](https://en.wikipedia.org/wiki/Blob_detection#The_hybrid_Laplacian_and_determinant_of_the_Hessian_operator_.28Hessian-Laplace.29) [Mikolajczyk & Schmid '01]
  - [Maximally stable extremal regions (MSER)](https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions) [Matas '02]
* Affine invariant feature detection:
  - [Hessian-Affine](https://en.wikipedia.org/wiki/Hessian_affine_region_detector) [Mikolajczyk & Schmid '04]
  - [Harris-Affine](https://en.wikipedia.org/wiki/Harris_affine_region_detector) [Mikolajczyk & Schmid '04]
  - Edge-based regions (EBR) [Tuytelaars & Van Gool '04]
  - Intensity-extrema-based regions (IBR) [Tuytelaars & Van Gool '04]
* Object recognition:
  - [Salient Regions](https://en.wikipedia.org/wiki/Kadir%E2%80%93Brady_saliency_detector) [Kadir & Brady '01]
* Feature description
  - [Scale-invariant feature transform (SIFT)](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) [Lowe '04]
  - [Speeded up robust features (SURF)](https://en.wikipedia.org/wiki/SURF) [Bay '06]
  - [Histogram of oriented gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) [Dalal '05]

... and [many others](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision))

We will go into a few of these in detail.

## Global Feature Descriptors 
Recall that a global feature descriptor describes the entire image as a whole, by extracting useful information and throwing away extraneous information.

## Histogram of Oriented Gradients (HOG) 
In the HOG feature descriptor, the features computed are the distribution (histograms) of directions of gradients (oriented gradients). As we've seen before, gradients ($$x$$ and $$y$$ derivatives) of an image are useful because the magnitude of gradients is large around edges and corners, and we know that edges and corners pack in a lot more meaning about an object shape than do flat regions.

There are several steps to computing HOG features, and we will go through them one by one.

### Step 1: Preprocessing
In the HOG paper, input$ images must have a fixed aspect ratio of $$1:2$$ and be resized to $$64 \times 128$$ pixels. Typically, we compute HOG features on multiple image patches at different locations and multiple scales. The only constraint is that the patch must have an aspect ratio of $$1:2$$ and be resized to $$64 \times 128$$ pixels. 

![](http://www.learnopencv.com/wp-content/uploads/2016/11/hog-preprocessing.jpg)
{: style="width:700px"}

Usually, with feature detectors, a key preprocessing step is to normalize the color values. However, as Dalal and Triggs point out, this step can be omitted in HOG descriptor computation, as the ensuing descriptor normalization essentially achieves the same result. 

So, once we crop and resize image patch, we're ready to calculate the HOG descriptor for this image patch.

### Step 2: Calculate the gradient images
To calculate a HOG descriptor, we need to first calculate horizontal and vertical gradients. This is achieved by filtering the image with the following kernels:

![](http://www.learnopencv.com/wp-content/uploads/2016/11/gradient-kernels.jpg)
{: style="width:400px"}

So, given an image $$I$$, we obtain the $$x$$ and $$y$$ derivatives by convolving the image with each filter:

$$ I_x = I * D_x $$

$$ I_y = I * D_y $$

where $$D_x$$ is the first kernel, and $$D_y$$ is the second kernel.

Given the $$x$$- and $$y$$-gradients for each pixel in the image, $I_{ij}$, we calculate the magnitude and orientation of the gradients:

$$ \theta_{I_ij} = tan^{-1} \Bigg( \frac{\partial I_{ij}}{\partial x} \big/ \frac{\partial I_{ij}}{\partial y} \Bigg) $$ 

$$ \lvert \nabla{I_{ij}} \lvert = \sqrt{ \Big( \frac{\partial I_{ij}}{\partial x} \Big)^2 + \Big( \frac{\partial I_{ij}}{\partial y} \Big)^2 {}_{}}$$

The figure below shows the gradients for our example image patch. The left image is the absolute value of the $$x$$-gradient. The center image is the absolute value of the $$y$$-gradient. The right image is the magnitude of the gradient.

![](http://www.learnopencv.com/wp-content/uploads/2016/11/gradients.png)
{: style="width:500px"}


### Step 3: Calculate histogram of gradients in 8x8 cells
In this step, the image is divided into 8x8 cells, and a histogram of gradients is calculated for each cell.

The histogram is essentially a vector of 9 bins discretizing the 180 possible gradient orientations (Dalal and Triggs use "unsigned" gradients, so we only need consider 180 degrees in this case. Empirically, in the original HOG paper, which was written for pedestrian detection, unsigned gradient orientations worked better.) The bins represent ranges of gradient orientations, e.g., 0-20 degrees, 20-40 degrees, all the way to 160-180 degrees.

![](http://www.learnopencv.com/wp-content/uploads/2016/12/hog-cell-gradients.png)
{: style="width:500px"}

A bin in the histogram is selected based on the gradient orientation, and the value that goes in the bin (we call this the "vote") is selected based on the gradient magnitude. The vote weight can be the gradient magnitude itself, or the squar root or square of the gradient magnitude.

For the above patch, our histogram looks like this:

![](http://www.learnopencv.com/wp-content/uploads/2016/12/histogram-8x8-cell.png)
{: style="width:500px"}
There is a lot of weight around 0 and 180 degrees, which means that most of the patch gradients are pointing up or down. 

### Step 4: 16x16 block normalization
Gradients of an image are sensitive to overall lighting. For example, if we make the image darker by dividing all pixel values by 2, the gradient magnitude will change by half, and therefore the histogram values will change by half. Ideally, we want our descriptor to be independent of lighting variations.

Therefore, we want our histograms to be normalized, so that it is not affected by lighting variations. 

Instead of just normalizing each 9x1 histogram, we normalize over a bigger sized block of $$16 \times 16$$. A $$16 \times 16$$ block has 4 histograms which can be concatenated to form a $$36 \times 1$$ element vector, which we then normalize. We then move the window by 8 pixels (see the gif below) and a normalized $$36 \times 1$$ vector is calculated over this window, and the process is repeated.

![](http://www.learnopencv.com/wp-content/uploads/2016/12/hog-16x16-block-normalization.gif)

### Step 5: Calculate the HOG feature vector
To calculate the final feature vector for the entire image patch, the $$36 \times 1$$ histograms are concatenated into one long vector. 

## Local Invariant Features
Local invariant features are a way to describe the constituent objects/parts in an image.

### Motivations
How do we identify the same point(s) in different images depicting the same thing, but which may have different orientations, lighting, occlusions, articulation etc? 

![](http://www.cc.gatech.edu/~hays/compvision/results/proj2/html/sshah426/matches.jpg)

How do we recognize the same **object class** in different images which may have wide intra-category variations (e.g., "wheel" in different images of motorcycles)?

![]({{site.baseurl}}/assets/images/motorcycles.png)
{: style="width:400px"}

### Properties 
Local invariant features have the following properties:
* **Local**: features are local, so they are robust to occlusion and clutter 
* **Invariant**, or covariant (we will discuss invariant transformations in the next section)
* **Robust**: noise, blur, discretization, compression, etc. do not have a big impact on the feature
* **Distinctive**: individual features can be matched to a large database of objects
* **Quantity**: we need a sufficient number of regions to cover the object 
* **Accurate**: precise localization
* **Efficient**: close to real-time performance

### Invariance

When a transformation is applied to an image, an **invariant** measure remains unchanged. 
A **covariant** measure changes in a way consistent with the image transformation.

Below are the levels of geometric invariance:
![]({{site.baseurl}}/assets/images/geometric-invariance.png)
{: style="width:600px"}

## Harris Corner Detector

## Laplacian of Gaussian (LoG)

The Laplacian of Gaussians and the [Difference of Gaussians](#difference-of-gaussians-dog) are both **blob detection** techniques. They detect regions in an image that differ in properties, such as brightness or color, compared to surrounding regions. Informally, a **blob** is a region of an image in which some properties are constant or approximately constant. Below is an image with several "blobs" circled. 

![](http://www.cs.utah.edu/~manasi/coursework/cs7960/p1/images/sunflower2_blob.png)

Blob detectors are different from edge or corner detectors because they provide complementary information about regions, whereas edge/corner detectors just provide the outline of a region of interest. Some uses for blob descriptors include object recognition and object tracking, image segmentation, and texture recognition. 

The Laplacian of Gaussians is named as such because we first apply a Gaussian filter to the original image, and then apply a Laplacian filter to the image after it has been convolved with the Gaussian filter.

### Step 1: Gaussian filter 

First, to smooth the image, we convolve the input image, $$f(x,y)$$, by a Gaussian kernel:

$$ G_t(x, y) = \frac{1}{2 \pi t} e^{-\frac{x^2 + y^2}{2t} } $$

at a certain scale $$t$$ to give a scale space representation $$ L_t(x,y) = G_t(x,y) * f(x,y). $$
(See [this section](#effects-of-noise) for a review of the Gaussian kernel).

We add Gaussian blur because the Laplacian (second-order derivative of the image) is extremely sensitive to noise. The blur helps smooth the image and stabilize the second-order derivative.

### Step 2: Laplacian filter

The **Laplacian filter**, $$\nabla^2$$, is just another linear filter. It is the Laplacian operator (essentially, the sum, over all dimensions, of the second-order gradient of each input dimension):

$$ \nabla^2 f(x,y) = \frac{\partial^2 f}{\partial x} + \frac{\partial^2 f}{\partial y} $$

For this step, we find the **zero-crossings** of the image, $$O(x,y)$$, using the Laplacian filter applied to the Gaussian-filtered image. The zero-crossings correspond to the positions in the input image of maximum gradient, and they can be used to localize edges.

$$ O(x,y) = \nabla^2( L_t(x,y)) = \nabla^2 \Big( f(x,y) * G_t(x,y) \Big) $$

Conveniently, this expression simplifies, so we only have to take the Laplacian of the Gaussian and then convolve that result with the input image:

$$ O(x,y) = \nabla^2 G_t(x,y) * f(x,y) $$

#### Taking the first and second derivative of a Gaussian

$$ G_t(x, y) = \frac{1}{2 \pi t} e^{-\frac{x^2 + y^2}{2t}} $$ 

Using the chain rule, we can compute the first-order derivative with respect to $$x$$:

$$ 
\begin{split}
\frac{\partial}{\partial x} G_t(x, y) &= \frac{1}{2 \pi t} 
\frac{\partial}{\partial x} e^{-\frac{x^2 + y^2}{2t}} \\
&= \frac{1}{2 \pi t} 
e^{-\frac{x^2 + y^2}{2t}} \frac{\partial}{\partial x} \Big( -\frac{x^2 + y^2}{2t} \Big) \\
&= \frac{-x}{t} \cdot \frac{1}{2 \pi t}
e^{-\frac{x^2 + y^2}{2t}} \\
&= \frac{-x}{2 \pi t^2} 
e^{-\frac{x^2 + y^2}{2t}} 
\end{split}
$$   

Now we can compute the second-order derivative with respect to $$x$$ (skipping a few steps, but using the product rule):

$$
\begin{split}
\frac{\partial^2}{\partial x} &= \frac{\partial}{\partial x} 
\Big( \frac{-x}{2 \pi t^2}
e^{-\frac{x^2 + y^2}{2t}} \Big) \\
&= \frac{ x^2 - t}{2 \pi t^3} e^{-\frac{x^2 + y^2}{2t}} 
\end{split}
$$

Below is a 3D visualization of the original Gaussian, its first and second derivatives, and their 2D projections. 

![]({{site.baseurl}}/assets/images/gaussian-derivs.png)

Below is an image before and after the Laplacian-of-Gaussians is applied to it:

<div style="display:inline-block;">
  <div style="float:left">
    <img src="https://softwarebydefault.files.wordpress.com/2013/05/monarch_in_may.jpg" style="height:200px"/>
    <p style="width:300px;text-align:center;font-size:14px">Before LoG</p>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="https://softwarebydefault.files.wordpress.com/2013/05/laplacian_3x3.jpg" style="height:200px"/>
    <p style="width:300px;text-align:center;font-size:14px">After LoG</p>
  </div>
</div>

As we can see, the white parts in the image are the zero-crossings (areas of maximum gradient), and the black parts are the areas of low gradient (little change).

## Difference of Gaussians (DoG) 

The Difference of Gaussians filter is an approximation of the Laplacian of Gaussians. The two algorithms are very similar, except, with DoG, instead of taking the Laplacian of the Gaussian filter, we take difference of two Gaussians (subtract them).

### Step 1: Gaussian filter

As before, we convolve the input image, $$f(x,y)$$, by a Gaussian kernel: 

$$ G_{t_1}(x, y) = \frac{1}{2 \pi t_1} e^{-\frac{x^2 + y^2}{2t_1} } $$

at a certain scale $$t_1$$ to give a scale space representation $$ L_{t_1}(x,y) = G_{t_1}(x,y) * f(x,y). $$

We do the same thing for a different with $$t_2$$ to get $$ L_{t_2}(x,y) = G_{t_2}(x,y) * f(x,y). $$

### Step 2: Take the difference of Gaussians

$$ O(x, y) = L_{t_1}(x,y) - L_{t_2}(x,y) = 
G_{t_1}(x,y) * f(x,y) - G_{t_2}(x,y) * f(x,y) =
\big( G_{t_1}(x,y) - G_{t_2}(x,y) \big) * f(x,y) 
$$

We set $$t_2 > t_1$$, such that the second Gaussian filter has more variance. Therefore, we essentially end up subtracting one blurred version of an original image from another, less blurred version of the original.

### What is DoG intuitively doing?

Blurring an image using a Gaussian kernel suppresses only high-frequency spatial information. 
Subtracting one image from the other preserves spatial information that lies between the range of frequencies that are preserved in the two blurred images. 
Thus, the difference of Gaussians is a band-pass filter that discards all but a handful of spatial frequencies that are present in the original grayscale image.

The Difference of Gaussians algorithm is believed to mimic how neural processing in the retina of the eye extracts details from images destined for transmission to the brain.

<div style="display:inline-block;">
  <div style="float:left">
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/da/Flowers_before_difference_of_gaussians.jpg" style="height:200px"/>
    <p style="width:200px;text-align:center;font-size:14px">Before DoG</p>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Flowers_after_difference_of_gaussians_grayscale.jpg/220px-Flowers_after_difference_of_gaussians_grayscale.jpg" style="height:200px"/>
    <p style="width:200px;text-align:center;font-size:14px">After DoG</p>
  </div>
</div>

## SIFT 
 
Earlier, we learned about corner detectors like the [Harris Corner Detector](#harris-corner-detector). These feature detectors are rotation-invariant, meaning that even if image is rotated, we will still detect the same corners.
However, what about scaling? For example, see the image below. 
When the window on an image is small, a corner may not look like a corner. So, the Harris Corner Detector is **rotation-invariant** but not **scale-invariant**.

![]({{site.baseurl}}/assets/images/sift-scaling.png)
{: style="width:400px"}

Scale-invariant feature transform (SIFT) is a 2004 local feature detector algorithm which extracts **image keypoints** and computes their descriptors. 

**Keypoints** are scale-invariant, rotation-invariant, circular "points of interest" in an image. They define what is interesting or what stands out in the image. No matter how the image changes (rotates, shrinks/grows, is subject to distortion), you should be able to find the same keypoints in the image. 
Below is an example of some image keypoints. 

![](https://i.stack.imgur.com/L4RUT.png)
{:style="width:600px"}

SIFT has several steps, which we will explore one by one.

### Step 1: Create the scale-space for an input image 

It is obvious that we can't use the same window to detect keypoints with a different scale. For this, we use **scale-space filtering**. The general idea is to create a grid of images at progressively smaller scales and larger amounts of blur. 
We add blur in order to intentionally get rid of some detail from the image, without introducing new false details. 

#### Octaves and Scales
We change the image scale and add blur for several "octaves" of the image. 
* The images in a single octave are the same size (scale), but have different amounts of blur.
* We linearly increase the amount of blur. So, if we start with a blur value of $$\sigma$$, then the amount of blur in the next image will be $$k \sigma$$, where $$k$$ is a constant that you choose. 
* We get from one octave to the next by halving the size of the image. 

The original SIFT algorithm suggests using 4 octaves and 5 blur levels. It also sets $$\sigma=1.6$$ and $$k=\sqrt{2}$$. Here's what that looks like on one image:

![](http://aishack.in/static/img/tut/sift-octaves.jpg)
{: style="width:500px"}

### Step 2: LoG Approximations
In the previous step, we created the scale-space of an image, creating a grid of progressively blurred images at progressively smaller sizes. In this step, we use those blurred images to generate another set of images, the [Difference of Gaussians](#difference-of-gaussians-dog). DoG acts as a blob detector which detects blobs in various sizes, so the DoG images will help us find interesting image keypoints. 

#### Why use DoG, not LoG?
In the [Laplacian of Gaussian](#laplacian-of-gaussian-log) (LoG) operation, we calculate second order derivatives (the "Laplacian") on a Gaussian-blurred image. This locates edges and corners on the image, which are good for finding keypoints. 

However, calculating all of those second-order derivatives is computationally expensive, so the SIFT algorithms uses the [Difference of Gaussians](#difference-of-gaussians-dog), which is an approximation of LoG. 

To generate the Difference of Gaussians quickly, we use the scale space that we generated in Step 1. We calculate the difference between images with different amounts of blur: 

![](http://aishack.in/static/img/tut/sift-dog-idea.jpg)
{: style="width:500px"}

### Step 3: Keypoint Localization
Finding image keypoints is a two-step process:

#### Step 3a: Locate maxima/minima in DoG images
The first step is to coarsely locate the maxima/minima (extrema) pixels in the DoG images. We do this by simply iterating over each pixel and checking all of its neighbors. The check is done within the current image and also the images above and below it in the scale-space. There are **26 neighbor checks** (8 in the current DoG image, 9 in the DoG image above, 9 in the DoG image below). X is marked as a keypoint if it is the greatest or least of all 26 neighbors.
This basically means that the keypoint is best represented at that scale. 

![](http://docs.opencv.org/3.1.0/sift_local_extrema.jpg)
{: style="width:500px"}

Note that keypoints are not detected in the lowermost and topmost scales. There simply aren't enough neighbors to do the comparison. So we ignore the DoG images at the lowermost and topmost scales.

#### Step 3b: Find subpixel maxima/minima 
Once the potential keypoint locations are found, they have to be refined to get more accurate results.
Initially, to localize keypoints, the authors of the SIFT paper just used the location and scale of the candidate keypoint.
The new approach calculates the interpolated location of the extremum, which substantially improves matching and stability.
The interpolation is done using the quadratic Taylor expansion of the relevant DoG image, $$D(x,y,\sigma)$$ with the candidate keypoint as the origin. This Taylor expansion is given by:

$$
D(\mathbf{x}) = D + \frac{\partial D^T}{\partial \mathbf{x}} \mathbf{x} 
+ \frac{1}{2} \mathbf{x}^T \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}
$$

where $$D$$ and its derivatives are evaluated at the candidate keypoint and $$\mathbf{x} = (x, y, \sigma)^T$$ is the offset from this point.

We can easily find the extreme points of this equation (differentiate and set $$=0$$). Each extreme point, $$\hat{\mathbf{x}}$$, is a subpixel keypoint location. 

The author of SIFT recommends generating two such extrema images. So, you need exactly 4 DoG images. To generate 4 DoG images, you need 5 Gaussian blurred images. This is why we need 5 level of blurs in each octave. 

### Step 4: Discarding low-contrast keypoints
The keypoints generated in the previous step produce a lot of keypoints. Some of them lie along an edge, or they don't have enough contrast, which makes them not useful as features. So, we discard them. 

#### Step 4a: Removing low contrast features
If the magnitude of the intensity at the current pixel in the DoG image (that is being checked for minima/maxima) is less than a certain value, it is rejected. 

Because we have subpixel keypoints (we used the Taylor expansion to refine keypoints), we again need to use the Taylor expansion to get the intensity value at subpixel locations. If its magnitude is less than a certain value ($$0.03$$ in the SIFT paper), we reject the keypoint.

#### Step 4b: Removing edges
The idea is to calculate two gradients at the keypoint, both which are mutually perpendicular. Based on the image around the keypoint, three possibilities exist. The image around the keypoint can be:
* **A flat region**: if this is the case, both gradients will be small.
* **An edge**: Here, one gradient will be large (the one that is perpendicular to the edge), and the other one will be small (parallel to the edge).
* **A corner**: Here, both gradients will be large.

Corners are great keypoints. So, we just want corners. If both gradients are large enough, we will accept it as a keypoint. Otherwise, it is rejected.

Mathematically, this is achieved by the Hessian Matrix (see the section on the [Harris Corner Detector](#harris-corner-detector). 

The figure below shows the detected keypoints on an image for Steps 3-4: 

<div style="display:inline-block;">
  <div style="float:left">
    <img src="{{site.baseurl}}/assets/images/sift-keypoints-1.jpg" style="height:250px"/>
    <p style="width:260px;text-align:center;font-size:14px">Original scale space extrema</p>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="{{site.baseurl}}/assets/images/sift-keypoints-2.jpg" style="height:250px"/>
    <p style="width:260px;text-align:center;font-size:14px">Low contrast keypoints are discarded</p>
  </div>
  <div style="float:left;margin-left:20px">
    <img src="{{site.baseurl}}/assets/images/sift-keypoints-3.jpg" style="height:250px"/>
    <p style="width:260px;text-align:center;font-size:14px">Keypoints on edges are discarded</p>
  </div>
</div>

### Step 5: Orientation Assignment
After Step 4, we have legitimate keypoints that have been tested to be stable. We already know the scale at which the keypoint was detected (it's the same as the scale of the blurred image). So, we already have **scale invariance**. The next thing is to assign an orientation to each keypoint. This orientation provides **rotation invariance**.

The general idea is to collect gradient directions and magnitudes around each keypoint (in a fixed window size). Then we determine the most prominent orientation(s) in that region, and we assign this orientation(s) to the keypoint. Any later calculations are done relative to this orientation. This ensures rotation invariance.

![](http://aishack.in/static/img/tut/sift-a-keypoint.jpg)
{: style="width:500px"}

Gradient magnitudes and orientations are calculated using pixel differences as follows:

$$
\lvert \nabla f \lvert = 
\sqrt{ \big[ L(x+1,y)-L(x-1,y) \big]^2 + \big[ L(x,y+1)-L(x,y-1) \big]^2 }
$$

$$
\theta(x,y) = \tan^{-1} \Big( \frac{L(x,y+1) - L(x,y-1)}{L(x+1,y) - L(x-1,y)}\Big)
$$

The magnitude and orientation is calculated for all pixels around the keypoint, where each neighbor pixel is indexed as $$(x,y)$$ and $$L(x,y)$$ returns the intensity of the pixel at that location. The window size, or "orientation collection region," is equal to $$1.5 \cdot \sigma_k$$, where $$\sigma_k$$ is the scale of the keypoint. 

We create a histogram, where the 360 degrees of orientation are broken into 36 bins (e.g., 0-9 degrees, 10-19 degrees, etc.). We put the gradient orientation for each pixel around the keypoint in this histogram, in an "amount" that is proportional to the magnitude of the gradient at that point weighted by Gaussian blur with $$\sigma = 1.5\sigma_k$$, where $$\sigma_k$$ is the scale of the keypoint. 

The histogram will have a peak at some orientation, and we assign the keypoint that orientation bin. 

Also, any peaks above 80% of the highest peak are converted into a new keypoint. This new keypoint has the same location and scale as the original, but its orientation is equal to the other peak. So, orientation can split up one keypoint into multiple keypoints:

![](http://aishack.in/static/img/tut/sift-orientation-histogram.jpg)
{: style="width:500px"}

### Step 6: Keypoint Descriptor
The final step of generating SIFT features is to create a uniquely identifying "fingerprint" for each keypoint. This fingerprint should be easy to calculate. It should also be relatively lenient when compared against other keypoints (because conditions are never *exactly* the same when comparing two different images). 

We start by creating a 16x16 neighborhood around the keypoint. It is divided into 16 sub-blocks of size 4x4. For each sub-block, we calculate gradient magnitudes and orientations and create an 8-bin orientation histogram. (So, there are a total of $$16 \times 8 = 128$$ bins available). The first bin has range 0-44 degrees, the second bin has range 45-89 degrees, and so on. 

![](http://aishack.in/static/img/tut/sift-fingerprint.jpg)
{: style="width=500px"}

As always, the amount added to the bin depends on the magnitude of the gradient.
However, in this step, the amount added to the histogram also depends on the distance of the keypoint.
This is done using a Gaussian weighting function. The further away the neighbor pixel is, the smaller the amount that gets added to the histogram.  

![](http://aishack.in/static/img/tut/sift-gaussian-4x4-weighting1.jpg)
{: style="width=500px"}

Once we have all 128 numbers (the 8 bins in 16 different histograms from the 16 different 4x4 sub-blocks), we normalize them, and these values form the **feature vector** that uniquely identifies the keypoint.

#### Solving a few last problems
* **Rotation dependence**. The feature vector uses gradient orientations (in the histograms). If we rotate the image, this feature vector no longer works. To achieve rotation invariance, the keypoint's rotation is subtracted from each orientation. Thus each gradient orientation is relative to the keypoint's orientation. 
* **Illumination dependence**. If we threshold values that are large, we can achieve illumination dependence. So, we threshold all of the 128 values to be $$<= 0.2$$. We then re-normalize the feature vector. Now our feature vector is illumination independent! 

### Step 7: Keypoint Matching
Now, how do we actually use these feature vectors for keypoints? In general, we use them to **match keypoints across multiple images.**

Keypoints between two images are matched by identifying their nearest feature vector neighbours (using the Euclidean distance metric or something similar). However, in some cases, the second closest-match may be very near to the first. It may happen due to noise or some other reasons. In that case, the ratio of closest-distance to second-closest distance is taken. If this ratio is greater than 0.8, then the keypoint match is rejected. This eliminates around 90% of false matches while discards only 5% correct matches, as per the paper.



## Resources:
* Professor Fei-Fei Li's CS 131 [lecture 1](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture1_introduction_cs131_2016.pdf), [lecture 4](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture4_pixels%20and%20filters_cs131_2016.pdf), and [lecture 5](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture5_edges_cs131_2016.pdf) slides.
* [OpenCV Canny Edge Detection tutorial](http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html)
* [Cornell CS 6670 lecture notes](http://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf)
* [Canny Edge Detection](http://www-scf.usc.edu/~boqinggo/Canny.htm)
* Tinne Tuytelaar's [ECCV tutorial on Local Invariant Features](http://homes.esat.kuleuven.be/~tuytelaa/tutorial-ECCV06.pdf)
* [Open CV Introduction to SIFT](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html)
* [Utkarsh Sinha's SIFT tutorial](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/)
* [Satya Mallick's Histogram of Oriented Gradients tutorial](http://www.learnopencv.com/histogram-of-oriented-gradients/)
