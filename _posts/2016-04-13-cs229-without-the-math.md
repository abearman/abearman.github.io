---
layout: default
title:  "CS229 without all the math"
date:   2016-04-13 12:00:00
categories: main
---

<head>
<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
</head>

# CS229 Without All the Math  

## 1. Supervised Learning 
Supervised learning is a class of machine learning where you're given fully-labeled training data, and you try to learn a classifier to label unseen data points. For example, let's say we have a dataset of houses, where we know the geolocation and square footage and we want to predict the housing price. In supervised learning, we take a set of fully labeled data (we know all three variables: geolocation, square footage, and price) and from this data learn a **hypothesis function** $$h$$ to predict housing price, for examples that are "unlabeled" (we know geolocation and square footage, but not price). 

### 1a. Linear Regression 
Linear regression is a ML algorithm used for a **regression** problem (where the input has continuous values, like the height of a person, or housing prices). Later, we'll see how to solve the classification problem (where the input has discrete values, such as whether an email is spam/not spam). 

#### Hypothesis function: $$\theta^T x$$
To perform supervised learning, we must decide how we’re going to represent functions/hypotheses $$h$$ in a computer. 
As an initial choice, let’s say we decide to approximate $$y$$ as a linear function of $$x$$:

$$
h(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 
$$

The $$\theta_i$$'s are the **parameters** (also called **weights**) parameterizing the space of linear functions mapping from $$\mathcal{X}$$ (the training features) to $$\mathcal{Y}$$ (the output labels). The $$x_i$$'s are the **training features**: here, $$x_1$$ is geolocation and $$x_2$$ is square footage. 

To simplify our notation, we also introduce the convention of letting $$x_0 = 1$$ (this is the **intercept term**), so we can write:

$$
h(x) = \sum_{i=0}^n \theta_i x_i = \theta^T x
$$ 

where $$\theta$$ and $$x$$ are now both vectors.

#### Cost function: least mean squares
Now, we need to define a **cost function** to judge how "good" the parameters $$\theta$$ we selected are. The cost function we'll use for linear regression is the least-squares function:

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^m \big( h_{\theta}(x^{(i)}) - y^{(i)} \big)^2
$$

#### Stochastic gradient ascent 
In order to optimize our parameters $$\theta$$ with this cost function, we need to perform gradient updates:

$$
\theta_j \gets \theta_j + \alpha \frac{\partial J}{\partial \theta_j} 
$$
for each parameters $$\theta_j$$.

To do this, we need to take the gradient of $$J(\theta)$$ with respect to $$\theta_j$$. We do this using **stochastic gradient ascent**,  i.e. one update for each training example $$x^{(i)}$$ (we can also do this with a small **batch** of training examples).

$$
\begin{split}
\frac{\partial J(\theta)}{\partial \theta_j} 
&= \frac{1}{2} \cdot \big( h_{\theta}(x^{(i)}) - y^{(i)} \big) \frac{\partial h_{\theta}}{\partial \theta}_j \\
&= \big( h_{\theta}(x^{(i)}) - y^{(i)} \big) \frac{\partial h_{\theta}}{\partial \theta_j} \\
&= \big( h_{\theta}(x^{(i)}) - y^{(i)} \big) x^{(i)}_j 
\end{split}
$$

Therefore, our gradient ascent looks like:

$$
\begin{split}
& \text{Repeat until convergence } \{ \\
& \hspace{1em} \text{for } i = 1:m \{ \\
& \hspace{2em} \text{for } j = 1:n \{ \\
& \hspace{3em} \theta_j \gets \theta_j + \alpha \big( h_{\theta}(x^{(i)}) - y^{(i)} \big) x^{(i)}_j \\
& \hspace{2em} \} \\
& \hspace{1em} \} \\
& \}
\end{split}
$$ 

Stochastic gradient ascent (gradient updates with just one training example or a small batch of training examples) is superior to full batch gradient ascent, because the latter algorithm has to scan through
the entire training set before taking a single step. This is a costly operation if $$m$$ is
large. Stochastic gradient ascent can start making progress right away, and
continues to make progress with each example it looks at.

### 1b. Logistic Regression
Let's now talk about the **classification problem**. This is just like the **regression
problem**, except that the values $$y$$ we now want to predict take on only
a small number of discrete values. For now, we will focus on the binary
classification problem in which $$y$$ can take on only two values, 0 and 1.


#### Hypothesis function: $$\sigma(\theta^T x)$$
We could approach the classification problem ignoring the fact that $$y$$ is
discrete-valued, and use our old linear regression algorithm to try to predict
$$y$$ given $$x$$. However, it is easy to construct examples where this method
performs very poorly. Intuitively, it also doesn’t make sense for $$h_{\theta}(x)$$ to take
values larger than 1 or smaller than 0 when we know that $$y \in \{0, 1\} {}_{}$$.

To fix this, let's change our hypothesis function to be the **logistic function** or **sigmoid function**:

$$
h_{\theta}(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}} {}_{}
$$

This is a graph of the sigmoid function, $$\sigma$$:

![Sigmoid function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

The sigmoid function squashes its input, $$z$$, to be between 0 and 1. As $$z \to \infty$$, $$\sigma(z) \to 1$$. Symmetrically, as $$z \to -\infty$$, $$\sigma(z) \to 0$$.  

Why use the sigmoid function for logistic regression? The sigmoid function is useful because it outputs values between 0 and 1, which can be interpreted as probabilities. Its derivative is easy to calculate (as we'll see in a bit), and it's differentiable at all values, which is necessary for performing gradient ascent. 

#### Cost function: cross-entropy 
As with linear regression, the next step after defining our hypothesis function is to come up with a cost function that measures how well our learned parameters $$\theta$$ do in predicting the outputs, by comparing $$h_{\theta}(x) {}_{}$$ with the ground truth labels $$y$$. We'll do a bit more math here than in the linear regression case to show how the cross-entropy cost function comes about.

Let's start by interpreting the outputs of our hypothesis function, $$h$$, as probabilities:

$$ P(y = 1 \mid x; \theta) = h_{\theta}(x) $$

$$ P(y = 0 \mid x; \theta) = 1 - h_{\theta}(x) $$  

This can be represented more compactly as:

$$
p(y \mid x; \theta) = (h_{\theta}(x))^y (1 - h_{\theta}(x))^{1 - y}
$$

because when $$y = 1$$ the second term just becomes 1, and when $$y = 0$$ the first term becomes 1. 

Assuming that the $$m$$ training examples were generated independently, we
can then write down the likelihood of the parameters as:

$$
\begin{split}
L(\theta) &= p(y; x, \theta) \\
&= \prod_{i=1}^m p(y^{(i)}; x^{(i)}, \theta) {}_{} \\
&= \prod_{i=1}^m (h_{\theta}(x^{(i)}))^{y^{(i)}} (1 - h_{\theta}(x^{(i)}))^{1 - y^{(i)}} {}_{} 
\end{split}
$$

To get rid of the product, we can maximize the log-likelihood. The log function is monotonically increasing, so it doesn't change our maximization problem. 

$$
\begin{split}
\ell(\theta) &= \log L(\theta) \\
&= \sum_{i=1}^m y^{(i)} \log h_{\theta}(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) {}_{}
\end{split}
$$

Now, how do we maximize the log-likelihood, $$\ell(\theta)$$ with respect to the parameters $$\theta$$? As before, we can use gradient ascent to make small updates to $$\theta$$, hill-climbing in the direction of the optimum (largest) value of $$\ell(\theta)$$. 

I will skip the derivations of the gradients $$\frac{\partial \ell}{\partial \theta_j}$$. One useful thing to know is that the derivative of the sigmoid function is $$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$. The result is:

$$
\frac{\partial \ell}{\partial \theta_j} =
(y - h_{\theta}(x)) x_j
$$

This gives us the stochastic gradient ascent update (for a single training example) of:

$$
\theta_j \gets \theta_j + \alpha (y^{(i)} - h_{\theta}(x^{(i)})) x^{(i)}_j)
$$

### Summary of Supervised Learning
* We use supervised learning when we have a fully labeled training set
* There are two forms of supervised learning problems: regression and classification
* Regression is when we're trying to predict a continuous-valued output (such as housing prices)
* Classification is when we're trying to predict a discrete-valued output (such as emails being spam/not spam)
* A common ML algorithm to solve regression problems is linear regression, where we minimize a least-squares cost function using stochastic gradient ascent.
* A common ML algorithm to solve classification problems is logistic regression, where we minimize the cross-entropy cost function using stochastic gradient ascent. 
