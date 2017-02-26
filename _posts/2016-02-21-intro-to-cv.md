---
layout: default
title:  "Intro to CV"
date:   2016-02-21 13:50:00
categories: main
---

<head>
<script type="text/javascript"
        src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
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

## Image filters

A image filter is used to form a new image whose pixel are a combination of the image's original pixel values. The **goals** of applying filters are to:
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

{% raw %}
$$\begin{split}
g[n, m] &= \frac{1}{9} \sum_{k=n-1}^{n+1} \sum_{l=m-1}^{m+1} f[k, l] \\ 
&= \frac{1}{9} \sum_{k=-1}^{1} \sum_{l=-1}^{1} f[n-k, m-l]
\end{split}$$
{% endraw %}

<div style="width:100%">
  <div style="margin: 0 auto; width:80%">
    <img src="https://i.stack.imgur.com/PnWe2.png" style="width:1000px"/>
  </div>
</div>

### Filter example #2: Image Segmentation

Image segmentation based on the threshold:

{% raw %}
$$g[n, m] 
\begin{cases}
255 & \text{ if } f[n, m] > 100\\
0 & \text{ otherwise }
\end{cases}$$
{% endraw %}

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}


### Resources:
* Professor Fei-Fei Li's [CS 131 lecture 1 slides](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture1_introduction_cs131_2016.pdf)
* Professor Fei-Fei Li's [CS 131 Lecture 4 slides](http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture4_pixels%20and%20filters_cs131_2016.pdf) 
