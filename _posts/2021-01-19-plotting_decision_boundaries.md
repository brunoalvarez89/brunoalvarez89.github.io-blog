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

To me, the available literature regarding this procedure tends to be poorly explained, and the associated code can be quite cryptic too (as an example, the official guidelines from scikit-learn can be found [here](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) and [here](https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html) ). This, together with the fact that other available sources tend to be slightly-modified, copy-pasted versions of the previously mentioned links ([here](https://towardsdatascience.com/easily-visualize-scikit-learn-models-decision-boundaries-dd0fb3747508), [here](https://gist.github.com/anandology/772d44d291a9daa198d4), [here](https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface) and [here](https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly)), and that other sources that massage a little bit more the examples tend to overlook the caveats of matching color palettes between different `matplotlib` functions as `scatter()` and `contourf()` ([here](https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07)), motivated me to understand the procedure better and to provide some code so we can all always come back here to pick the right function to do the work :P
