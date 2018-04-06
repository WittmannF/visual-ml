# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_X_grid(X, values, cols):
    """
    Creates a dataframe to be used as input in the classifier for mapping 
    all the values between the minimum and maximum values of a feature. 

    Parameters
    ----------
    X : pandas dataframes
        Set of features 

    values : numpy array
        Either an 1D or 2D array with the values in which the decision boundary
        is going to be mapped

    cols : list
        one or two columns from X.columns that are going to be mapped
        IMPORTANT: when it is 1D, cols should be declared without brackets

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> random_values = np.random.rand(4,4)
    >>> X = pd.DataFrame(random_values, columns=['A','B','C','D'])
    >>> x = [1, 3, 4]
    >>> create_X_grid(X, x, 'D')
         A    B    C  D
    0  0.0  0.0  0.0  1
    1  0.0  0.0  0.0  3
    2  0.0  0.0  0.0  4
    >>> y = np.ones((2,5))
    >>> y
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    >>> create_X_grid(X, y, ['B','D'])
         A    B    C    D
    0  0.0  1.0  0.0  1.0
    1  0.0  1.0  0.0  1.0
    2  0.0  1.0  0.0  1.0

    """
    
    n_rows = len(values)
    n_columns = len(X.columns)
    X_grid = pd.DataFrame(np.zeros((n_rows,n_columns)), columns=X.columns)
    X_grid[cols] = values

    return X_grid

def raise_error(message, type=None):
    print(message)
    pass


def get_mesh_coordinates(clf, X, y, cols):
    """
    Takes either one feature or a pair of features and creates a grid 
    ranging 10 steps between their minimum and maximum values and then get the 
    classifier's output for each of these values in the grid (Z value)

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=42)
    >>> from sklearn.linear_model import LogisticRegression
    >>> clf = LogisticRegression(random_state=42).fit(X,y)
    >>> import pandas as pd
    >>> X = pd.DataFrame(X, columns=['A','B','C','D'])
    >>> xx, yy, Z = get_mesh_pdcoordinates(clf, X, y, 'A')
    >>> import matplotlib.pyplot as plt
    >>> plt.contourf(xx,yy,Z)




    """

    # First, check if we will map one or two features (1D or 2D)

    if len(cols) > 2:
        # Later I will replace this to a raise error function
        raise_error("Maximum number of input features exceeded. 'col' should \
            have either one value (1D) or two (2D)")
    elif len(cols)==2:
        # 2D coordinates
        pass
    else:
        # 1D coordinates
        n_steps = 11 # the value is 11 since there's 11 values in [0,10]
        min_x, max_x = np.min(X[cols]), np.max(X[cols])
        delta_x = max_x - min_x
        x = np.linspace(min_x, max_x, n_steps)
        y = np.linspace(0, delta_x, n_steps)

        xx, yy = np.meshgrid(x, y)

        # Create matrix if features that are going to be mapped
        X_grid = create_X_grid(X, xx.ravel(), cols)

        # Get prediction values (either probabilities or from decision function)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_grid)
        else:
            Z = clf.predict_proba(X_grid)[:, 1]

        Z = Z.reshape(xx.shape)

        return xx, yy, Z







    


