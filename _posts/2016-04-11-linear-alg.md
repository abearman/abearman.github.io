---
layout: default
title:  "Linear Algebra Review"
date:   2016-04-11 12:00:00
categories: main
---

<head>
<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
</head>   
         
# Linear Algebra Review

## Table of Contents
* [Eigenvectors and Eigenvalues](#eigenvectors-and-eigenvalues)
* [Singular Value Decomposition](#singular-value-decomposition)
* [Principle Components Analysis](#principal-components-analysis)

## Eigenvectors and Eigenvalues
### Definition
Linear equations have the form:

$$Ax = b$$

where $$A \in \mathbb{R}^{m \times n}$$ is a matrix, $$x \in \mathbb{R}^n$$ is a vector, and $$b \in \mathbb{R}^m$$ is a vector. $$A$$ essentially maps $$x$$ from a $$n$$-dimensional space into a $$m$$-dimensional space. $$A$$ acting on a vector $$x$$ does two things to $$x$$:
1. It scales the vector
2. It rotates the vector

However, for any matrix $$A$$, there are some *favored vectors and directions.* When the matrix acts on these favored vectors, the action just results in scaling the vector. There is no rotation:

$$ Ax = \lambda x $$ 

The scalar $$\lambda$$ is an **eigenvalue** of $$A$$. The vector $$x$$ is an **eigenvector** of $$A$$.
$$\lambda$$ tells us how much the vector $$x$$ is stretched, shrunk, reversed, or left unchanged when it is multipled by $$A$$.

### Special cases: $$\lambda = 0$$ and $$A = I$$
If the eigenvalue $$\lambda = 0$$, then the vector $$x$$ is in the **nullspace** of $$A$$.

If $$A$$ is the identity matrix, $$I$$, then every vector has $$Ax = x$$. All vectors are eigenvectors of $$I$$, and all eigenvalues are $$\lambda = 1$$. 

### Intuition
What does this mean intuitively? We can think of eigenpairs (eigenvector and its eigenvalue) as similar to roots of a polynomial. The roots of a polynomial *ground* the polynomial, limiting its shape. By knowing a polynomial's roots and degree, we can sketch the graph pretty well. 

Similarly, each eigenvector is like a skewer which helps hold the linear transformation in place. 

Consider the eigenvector corresponding to the maximum eigenvalue. If we take a vector along this vector, then the action of the matrix is maximum. ***No other vector when acted on by this matrix will get stretched as much as this eigenvector.***  

The eigenvectors are vectors which remain pointing along the same line they originally pointed along after being transformed by the matrix $$A$$.

## Singular Value Decomposition
### Definition
Singular value decomposition is essentially trying to reduce a rank $$r$$ matrix to a rank $$k$$ matrix, where $$k < r$$.

To find a SVD of a matrix $$A$$, we must find $$V$$, $$\Sigma$$, and $$U$$ such that:

$$ A = U \Sigma V^T $$

such that:
* $$V$$ must diagonalize $$A^T A$$, and $$v_i$$ are eigenvectors of $$A^T A$$
* The diagonal elements of $$\Sigma$$ are singular values of $$A$$.
* $$U$$ must diagonalize $$AA^T$$, and $$u_i$$ are eigenvectors of $$A A^T$$. 

If $$A$$ has rank $$r$$, then:
* $$v_1, \dots, v_r$$ forms an orthonormal basis for the range of $$A^T$$
* $$u_1, \dots, u_r$$ forms an orthonormal basis for the range of $$A$$
* $$rank(A)$$ is equal to the number of nonzero entries of $$\Sigma$$.

It can be shown that $$A$$ can be written as a sum of rank = 1 matrices:

$$ A = \sum_{i=1}^r \sigma_i u_i v_i^T $$

We know that $$\sigma_i$$ is monotonically decreasing, so the significance of each term decreases as $$i$$ increases. This means that the summation to $$k < r$$ is an approximation $$\hat{A}$$ of rank $$k$$ for the matrix $$A$$. SVD means we can take a list of $$r$$ unique vectors and approximate them as a linear combination of $$k$$ unique vectors. 

### Intuition
As before, we know that $$A$$ is a linear map from a $$n$$-dimensional vector space to a $$m$$-dimensional one: $$Ax = b$$. 

But we can also use $$A$$ as being a list of data points, so each row of $$A$$ is a datapoint in $$\mathbb{R}^n$$, and there are $$m$$ total data points, which are observations of some process happening in the world.

How do we reconcile these two different interpretations? There's some cognitive dissonance here. 

The way these two ideas combine is that the data is seen as the *image* of the basis vectors of $$\mathbb{R}^n$$ under the linear map specified by $$A$$. Let's break that down. Let's say I want to express people rating movies. Each row will correspond to the ratings of a movie, and each column will correspond to a person:

$$
\begin{array}{c|ccc}
& \text{Alex} & \text{Bob} & \text{Carrie} \\ \hline
\text{Up} & 2 & 5 & 3 \\ 
\text{Skyfall} & 1 & 2 & 1 \\ 
\text{Thor} & 4 & 1 & 1 \\ 
\text{Amelie} & 3 & 5 & 2 \\
\text{Casablanca} & 4 & 5 & 5 \\
\end{array}
$$ 

So, $$A$$ is a $$5 \times 3$$ matrix. The basis vectors of the $$\mathbb{R}^3$$ domain are called *people*. The basis vectors of the $$\mathbb{R}^5$$ codomain are *movies*. 

$$ 
\text{span} \{ \vec{e}_\text{Alex}, \vec{e}_\text{Bob}, \vec{e}_\text{Carrie} \} 
\to A \to
\text{span} \{ \vec{e}_\text{Up}, \vec{e}_\text{Skyfall}, \dots, \vec{e}_\text{Casablanca} \}
$$ 

The dataset is represented by 
$$ A \vec{e}_\text{Alex}, A \vec{e}_\text{Bob}$$, and $$A \vec{e}_\text{Carrie} {}_{}$$. 

This is useful, because we can start to see the modelling assumptions of linear algebra. If we're tyring to say something about how people rate movies, we would need to represent that person *as a linear combination of Alex, Bob, and Carrie.* Likewise, if we had a new movie and wanted to use the matrix to say anything about it, we'd have to represent the movie as a linear combination of the existing movies.

Now we get to the key: factorizing the matrix via SVD provides an alternative and more useful way to represent the process of people rating movies. By changing the basis of one or both vector spaces involved, we isolate the different (orthogonal) characteristics of the process. In the context of our movie example, ``factorization" means the following:
1. Come up with a special list of vectors $$v_1, v_2, \dots, v_5$$ so that every movie can be written as a linear combination of the $$v_i$$.
2. Do the analogous thing for people to get $$p_1, p_2, p_3$$.
3. Do steps 1 and 2 in such a way that the map $$A$$ is diagonal with respect to both bases simultaneously.  

## Principle Components Analysis
### Tldr
In one sentence, PCA can supply the user with a lower-dimensional picture of the data, a "shadow" of the object when viewed from its most informative viewpoint.

### Overview
The principal components are the directions of the data which have the most variance. They are in decreasing order, e.g., the first principal component has the most variance, the second principal component has the second most variance, and so on. See the graph below for an example.  

![PCA oval](http://weigend.com/files/teaching/stanford/2008/stanford2008.wikispaces.com/file/view/pca_example.gif)

Note that the principal component directions are orthogonal to each other. 

Going back to eigenvectors and eigenvalues, we recall that an eigenvector specifies a direction of a line (e.g., vertical, horizontal, 45 degrees). Another interpretation of an eigenvalue is a number telling you **how much variance there is in the data in the direction of the eigenvector.** The eigenvector with the highest eigenvalue is therefore the first principal component.

The number of eigenvectors/values in a dataset depends on the number of dimensions that the dataset has. Suppose we're measuring housing location and housing price -- there are 2 variables, so we have 2 eigenpairs. If we're measuring housing location, housing price, and square footage, then there are 3 eigenpairs.

However, the eigenvector directions are **not** the same as the original dimensions of the dataset. Consider the visualization below:

![PCA figure 1](https://georgemdallas.files.wordpress.com/2013/10/pca2.jpg)
{: style="width:400px"}

Let's say our original dataset has 2 dimensions: the $$x$$-axis is housing location and the $$y$$-axis is housing price. The graph below shows the two principal component directions: 

![PCA figure 2](https://georgemdallas.files.wordpress.com/2013/10/pca11.jpg)
{: style="width:400px"}

We can now reframe the data using these new dimensions:

![PCA figure 3](https://georgemdallas.files.wordpress.com/2013/10/pca1.jpg)
{: style="width:400px"}

We haven't changed the data; we're just projecting it into a new basis $$\{ x_\text{new}, y_\text{new} \}$$ instead of $$\{x, y\}$$. 

### Aside: Change of basis 
We've talked about needing to "project data into a new basis," but what does this mean and how do we do it? 

Recall that a **basis** for a vector space of dimension $$n$$ is a set of $$n$$ vectors $$(\alpha_1, \dots, \alpha_n)$$, called basis vectors, with the property that every vector in the space can be expressed as a unique linear combination of the basis vectors. 

The **standard basis** in $$\mathbb{R}^3$$ is the Cartesian coordinates, $$x$$, $$y$$, and $$z$$. In vector notation, we write:

$$ 
\left(
\begin{array}{c}
x \\
y \\
z  
\end{array} 
\right) = 
x \left(
\begin{array}{c}
1 \\
0 \\
0 
\end{array}
\right) + 
y \left(
\begin{array}{c}
0 \\
1 \\
0
\end{array}
\right) +
z \left(
\begin{array}{c}
0 \\
0 \\
1
\end{array}
\right) =
x \mathbf{e_1} + y \mathbf{e_2} + z \mathbf{e_3}
$$

where $$\{ \mathbf{e_1}, \mathbf{e_2}, \mathbf{e_3} \}$$ are the standard basis vectors.

Sometimes, we need to transform data from one basis to another, like above in PCA. This is called a **change of basis.**

### Application: dimension reduction
What is PCA used for? Primarily, it's for **dimension reduction.** This means reducing the data into its basic components, stripping away any noise.

We find the eigenvectors/values of the dataset (sorted in descending order), and choose all the pairs that have non-zero eigenvalues, or just choose the $$n$$ pairs with the largest eigenvalues. Then, we project the data into the basis defined by these eigenvectors. 

### Mathematical interpretation: the relationship of SVD to PCA
Read this section if you want a more rigorous explanation of how to do PCA, and how SVD is a general case of PCA.

Let's say our data matrix is called $$X$$, where columns represent the dimensionality (or number of features) in the dataset, and each row is a different data example. Let's say that $$X$$ is zero-centered, i.e. the column means have been subtracted from every column. In PCA, we must compute the eigenvectors/values of the covariance matrix, $$C = X^T X$$. The covariance matrix is symmetric, so it is diagonalizable:

$$
C = X^T X = V L V^T
$$  

where $$V$$ is a matrix of eigenvectors (each column is an eigenvector) and $$L$$ is a diagonal matrix with eigenvalues $$\lambda_i$$ in decreasing order on the diagonal. 

As before, the eigenvectors are the principal directions of the data (directions of largest variance, in descending order of more variance to less variance). The coordinates of the $$i$$-th data point in the new PCA space are given by the $$i$$-th row of $$X V$$.  

If we perform the singular value decomposition of $$X$$, we get:

$$
X = U \Sigma V^T
$$

We can rewrite the covariance matrix $$C$$ using the singular value decomposition of $$X$$:

$$ 
C = X^T X =  (U \Sigma V^T)^T U \Sigma V^T = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T 
$$

because $$U$$ is orthogonal, so $$U^T = U^{-1}$$ and therefore $$U^T U = U^{-1} U = I$$. 

Therefore, we can easily see that the matrix $$L$$ of eigenvalues from PCA is equal to $$\Sigma^2$$.

Principal components are given by $$X V = U \Sigma V^T V = U \Sigma$$.

### References
* [Jeremy Kun's SVD article](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)
* [George Dallas's article on PCA](https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/) 
