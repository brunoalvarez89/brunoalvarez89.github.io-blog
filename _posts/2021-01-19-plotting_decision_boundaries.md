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
$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$

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

The main idea is to use our classifier (in this case, a trained K-means clustering algorithm) to predict a **grid** from the input space. Since our input data lies on 2 dimensions, our grid is going to be 2-dimensional ($R^2$).

Let's start with importing stuff and then defining the minimum and maximum $(x,y)$ pairs of our grid (these will correspond to the lower-left and upper-right vertices of the grid/rectangle). To do this, we will just extract the minimum and maximum $x$ and $y$ values from our input data.

```python
import numpy as np

x_min = np.min(X[:,0])
y_min = np.min(X[:,1])
x_max = np.max(X[:,0])
y_max = np.max(X[:,1])

(x_min, y_min), (x_max, y_max)
```
