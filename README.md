# VisualML: Visualization of machine learning models

Visual ML is a library for visualizing the decision boundary of multi-dimensional machine learning models from sklearn using 2D projections of pairs of features. Here's an example:

```
import visualml as vml
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier as RF

# Create a toy classification dataset
feature_names = ['A','B','C','D']
X, y = make_classification(n_features=4, random_state=42)

# The visualization is only supported if X is a pandas df
X = pd.DataFrame(X, columns=feature_names)

# Train a classifier
clf = RF(random_state=42).fit(X,y) 

# Plot decision boundary grid
vml.decision_boundary_grid(clf, X, y)
```

![screen shot 2018-04-14 at 08 23 07](https://user-images.githubusercontent.com/5733246/38765430-f2599e5e-3fc9-11e8-9b26-8f783d35fece.png)
