---
title: "Plotting decision boundaries"
image:
  path: /assets/img/posts/decision_boundaries.jpg
  thumbnail: /assets/img/posts/decision_boundaries.jpg
categories:
  - Machine Learning
tags:
  - Plotting
---

When working with classification problems in Machine Learning, a common (and very useful) thing to do is to plot the decision boundaries of our classifiers. This helps us to discern which regions of our feature space are going to be assigned a given class.

To me, the available literature regarding this procedure tends to be poorly explained, and the associated code can be quite cryptic too (as an example, the official guidelines from scikit-learn can be found [here](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) and [here](https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html) ). This, together with the fact that other available sources tend to be slightly-modified, copy-pasted versions of the previously mentioned links ([here](https://towardsdatascience.com/easily-visualize-scikit-learn-models-decision-boundaries-dd0fb3747508), [here](https://gist.github.com/anandology/772d44d291a9daa198d4), [here](https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface) and [here](https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly)), and that other sources that massage a little bit more the examples tend to overlook the caveats of matching color palettes between different `matplotlib` functions as `scatter()` and `contourf()` ([here](https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07)), motivated me to understand the process better and provide some code so we all can always come back here to pick the right function to do the work :)

So, let's start. Since we are going to be playing around with classifiers, let's first generate some sample input data in 2 dimensions. For this, we are going to use the function `make_blobs()` from the `skearn.datasets` package, to which we are going to ask to create five 2-dimensional blobs (or centroids) of 200 points each

```python
from sklearn.datasets import make_blobs

n_blobs = 5
n_features = 2
n_samples = n_blobs*200

X, _ = make_blobs(n_samples=n_samples, centers=n_blobs, n_features=n_features, random_state=10)
```

As we can see, we have now some collection of 2-dimensional points

```
>>> X[:10]
array([[  4.4016599 ,  -9.42456185],
       [ -6.70699928,  -8.43356401],
       [ -6.25332117,   5.01582549],
       [  7.41179861,  -8.13779541],
       [ -4.83920837,   5.6893035 ],
       [  0.62620427,  -3.76598107],
       [ -1.5348865 ,  -5.02692144],
       [  0.45184509,  -3.86956441],
       [  6.25341149, -11.53947313],
       [ -8.3710159 ,  -7.91311993]])
```

Let's visualize these blobs as a scatter plot

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8), facecolor="white")
plt.scatter(X[:,0], X[:,1], alpha=0.85, s=150, edgecolor="black")
plt.title("Sample data", size=30)
plt.tight_layout()
```

![sample_data.jpg](/assets/img/posts/sample_data.jpg)

Now we are going to cluster this data using the [K-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering), provided in the `sklearn.cluster` package. This will assign each blob a proper class or label. For this, we are just going to use our known quantity of blobs (5) as the value of $$k$$.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=n_blobs, random_state=10).fit(X)
labels = kmeans.labels_
```
```
>>> labels[:10]
array([0, 4, 1, 0, 1, 2, 2, 2, 0, 4], dtype=int32)
```

Let's see how the clustering looks like

```python
plt.figure(figsize=(15,8), facecolor="white")
plt.scatter(X[:,0], X[:,1], c=labels, cmap="Set3", alpha=0.85, s=150, edgecolor="black")
plt.title("Sample data (clustered)", size=30)
plt.tight_layout()
```

![sample_data_clustered.jpg](/assets/img/posts/sample_data_clustered.jpg)

Let's work now on building our decision boundaries plot.

The main idea is to use our classifier (in this case, a trained K-means clustering algorithm) to predict a **grid** from the input space. Since our input data lies on 2 dimensions, our grid is going to be 2-dimensional ($$R^2$$).

Let's start with importing stuff and then defining the minimum and maximum $$(x,y)$$ pairs of our grid (these will correspond to the lower-left and upper-right vertices of the grid/rectangle). To do this, we will just extract the minimum and maximum $$x$$ and $$y$$ values from our input data

```python
import numpy as np

x_min = np.min(X[:,0])
y_min = np.min(X[:,1])
x_max = np.max(X[:,0])
y_max = np.max(X[:,1])
```
```
>>> (x_min, y_min), (x_max, y_max)
((-9.151387648413154, -12.564557783873104),
 (8.106323173314234, 8.003780005710317))
```

The first step to generate our rectangular grid is to define the sides of our rectangle. For this, we will extend a little bit beyond our minima and maxima (5%), just to avoid the problem of having data points plotted exactly on our grid boundaries. Here we also need to specify the resolution of our grid in terms of a sampling interval (lower intervals imply higher resolutions); we will use an interval of 0.05.

```python
extra_margin = 0.05
sampling_interval = 0.05

x_axis = np.arange(x_min*(1+extra_margin), x_max*(1+extra_margin), sampling_interval)
y_axis = np.arange(y_min*(1+extra_margin), y_max*(1+extra_margin), sampling_interval)
```
```
>>> x_axis.shape, y_axis.shape
((363,), (432,))
```

As we can see, for our example, the $$x$$ axis was partitioned in 363 chunks, while the $$y$$ axis in 432. 

We will now construct the 2D grid. To do so, we will use the `meshgrid()` function from `numpy`, which essentially maps all $$x$$ and $$y$$ values from its inputs (in our case, `x_axis` and `y_axis`) to their corresponding positions in the output grid. Following is an excellent figure explaining this ([credit goes to the user Sarsaparilla](https://stackoverflow.com/a/42404323/3368529))

![meshgrid.png](/assets/img/posts/meshgrid.png)

```python
x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
```
```
>>> x_mesh.shape, y_mesh.shape
((432, 363), (432, 363))
```

Now we need to feed this to our K-means. The problem is that the mesh format is not the appropriate one to feed to the algorithm! We need $$(x,y)$$ pairs, so let's construct them.

First, we flatten our meshes into 1-dimensional arrays

```python
x_mesh_flatten = x_mesh.flatten()
y_mesh_flatten = y_mesh.flatten()
```

Then, we reshape them to column vectors

```python
x_mesh_flatten_reshape = x_mesh_flatten.reshape((x_mesh_flatten.shape[0],1))
y_mesh_flatten_reshape = y_mesh_flatten.reshape((y_mesh_flatten.shape[0],1))
```

And then we stack them together, one next to the other

```python
X_grid = np.hstack((x_mesh_flatten_reshape,y_mesh_flatten_reshape))
```
```
>>> X_grid.shape
(156816, 2)
```

As we can see, we have all the grid in the format we need to run our predictions (156816 observations, 2 dimensions each).

Let's predict now

```python
labels_grid = kmeans.predict(X_grid)
```

Now, to plot the resulting classes, we will reshape the output to the same shape that our meshes have

```python
labels_mesh = labels_grid.reshape(x_mesh.shape)
```

And now everything is square. 

We can proceed to plot our boundaries using the `pcolormesh()` function (Note: before using this function, I tried for hours to make `contourf()` work properly with the colormaps, but I just could not do it, so it opted for `pcolormesh()`).

```python
plt.figure(figsize=(15,8), facecolor="white")
plt.pcolormesh(x_mesh, y_mesh, labels_mesh, cmap="Set3", alpha=0.1, shading="gouraud", zorder=0)
plt.tight_layout()
```

![decision_boundaries.jpg](/assets/img/posts/decision_boundaries.jpg)


