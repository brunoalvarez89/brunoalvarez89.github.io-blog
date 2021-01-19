---
title: "Plotting Decision Boundaries"
image:
  path: /assets/img/posts/decision_boundaries.jpg
  thumbnail: /assets/img/posts/decision_boundaries.jpg
categories:
  - Machine Learning
tags:
  - Plotting
---

When working with classification problems in Machine Learning, a common (and very useful) thing to do is to plot the decision boundaries of our classifiers. This helps us to discern which regions of our feature space are going to be assigned a given class.

To me, the available literature regarding this procedure tends to be poorly explained, and the associated code can be quite cryptic too (as an example, the official guidelines from scikit-learn can be found [here](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) and [here](https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html) ). This, together with the fact that other available sources tend to be slightly-modified, copy-pasted versions of the previously mentioned links ([here](https://towardsdatascience.com/easily-visualize-scikit-learn-models-decision-boundaries-dd0fb3747508), [here](https://gist.github.com/anandology/772d44d291a9daa198d4), [here](https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface) and [here](https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly)), and that other sources that massage a little bit more the examples tend to overlook the caveats of matching color palettes between different `matplotlib` functions as `scatter()` and `contourf()` ([here](https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07)), motivated me to understand the process better and provide some code so we can all always come back here to pick the right function to do the work :)

So, let's start. Since we are going to be playing around with classifiers, let's first generate some sample input data in 2 dimensions. For this, we are going to use the function `make_blobs()` from the `skearn.datasets` package, to which we are going to ask to create five 2-dimensional blobs (or centroids) of 200 points each

```python
from sklearn.datasets import make_blobs

n_blobs = 5
n_features = 2
n_samples = n_blobs*200

X, _ = make_blobs(n_samples=n_samples, centers=n_blobs, n_features=n_features, random_state=10)
```
As we can see, we have now some collection of 2-dimensional points

```python
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
