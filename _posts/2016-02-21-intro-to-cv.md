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

The **goal** of computer vision is to bridge the gap between pixels and "meaning." When you give a computer an image, all it sees is a 2D (or 3D, if the image is in color) numerical array:

<img src="http://images.slideplayer.com/16/5003478/slides/slide_7.jpg" style="width:500px;"/>

## What kind of information can we extract from an image? 

**Metric Information**

* 3D modeling
* Structure from motion
* Shape and motion capture
* etc.

<img src="http://www.cs.cornell.edu/projects/disambig/img/disambig_cover.png" style="width:700px;"/>

**Semantic Information**

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

## Image Filters

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

### Filter example #1: Moving Average

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

### Filter example #2: Image Segmentation

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

## Convolution

Convolution is the process of adding each element of an image to its local neighbors, weighted by a [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution) (essentially, a small matrix used to apply effects to an image, such as sharpening, blurring, or outlining). It is **not** traditional matrix multiplication. 

[This article](http://setosa.io/ev/image-kernels/) has great visuals to help explain image kernels.
### Example convolutions

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

## Edge Detection

We know that edges are special from vision studies ([Hubel & Wiesel, 1960s](http://hubel.med.harvard.edu/papers/HubelWiesel1964NaunynSchmiedebergsArchExpPatholPharmakol.pdf)). Edges encode most of the semantic and shape information of an image.  

The **goal** of edge detection is to identify sudden changes (edges) in an image. Ideally, we want to recover something like an artist line drawing.

<img src="http://www.clipartbest.com/cliparts/9iz/o4b/9izo4bRET.png" style="width:200px;"/>

### What causes edges?

<img src="https://image.slidesharecdn.com/finalminorprojectppt-140422115839-phpapp02/95/fuzzy-logic-based-edge-detection-11-638.jpg?cb=1398168182" style="width:1000px;"/>

### How do we characterize edges?

**Definition**: An edge is a place of rapid change in the image intensity function. Edges correspond to the extrema of the first derivative. 

<img src="https://mipav.cit.nih.gov/pubwiki/images/1/11/EdgeDetectionbyDerivative.jpg" style="width:200px"/>

### Image gradients

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

### Effects of noise

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

### Problems with this simple edge detection 

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
  </div><br>
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
  </div><br>
</div>

We can do all of these steps in one line in Python:

{% highlight python%}
import cv2
import numpy as np

img = cv2.imread('img_name.jpg', 0)  # Loads an image in grayscale
edges = cv2.Canny(img, 100, 200)  # 2nd and 3rd args are minVal and maxVal, respectively
{% endhighlight %}

### Resources:
* Professor Fei-Fei Li's CS 131 [lecture 1](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture1_introduction_cs131_2016.pdf), [lecture 4](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture4_pixels%20and%20filters_cs131_2016.pdf), and [lecture 5](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture5_edges_cs131_2016.pdf) slides.
* [OpenCV Canny Edge Detection tutorial](http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html)
* [Cornell CS 6670 lecture notes](http://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf)
* [Canny Edge Detection](http://www-scf.usc.edu/~boqinggo/Canny.htm)
