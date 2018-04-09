# -*- coding: utf-8 -*-
from __future__ import division
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def decision_boundary_grid(clf, X, y):
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

    """
    ### Plot main diagonal
    # Get the number of columns (features)
    n_cols = len(X.columns)
    fig, ax = plt.subplots(n_cols, n_cols)

    # Loop over subplots
    for i, col in enumerate(X.columns):
        plot_decision_boundary(clf, X, y, col, ax=ax[i][i])

    ### Plot off diagonal
    for i_x, col_x in enumerate(X.columns):
        for i_y, col_y in enumerate(X.columns):
            if i_x != i_y:
                plot_decision_boundary(clf, X, y, [col_x, col_y], \
                    ax=ax[i_x][i_y])


    plt.show()



def plot_decision_boundary(clf, X, y, cols, cm=plt.cm.RdBu, ax=None):
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
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=42)
    >>> from sklearn.svm import SVC
    >>> clf = SVC(random_state=42).fit(X,y)
    >>> import pandas as pd
    >>> X = pd.DataFrame(X, columns=['A','B','C','D'])
    >>> plot_decision_boundary(clf, X, y, 'A')

    """
    # Check if X is a DataFrame
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    if not isinstance(X, pd.DataFrame):
        raise_error("The input set of features X should be a Pandas DataFrame")
        return

    if len(cols) > 2:
        # Later I will replace this to a raise error function
        raise_error("Maximum number of input features exceeded. 'col' should \
            have either one value (1D) or two (2D)")

    elif len(cols)==2: # 2D plot (scatter)
        xx, yy, Z = get_mesh_coordinates(clf, X, y, cols)
        cm = plt.cm.RdBu
        if ax==None:
            plt.contourf(xx,yy,Z, cmap=cm)
            plt.scatter(X[cols[0]], X[cols[1]], c=y, cmap=cm_bright, \
                edgecolors='k', alpha=.7)
            ax=plt.gca()
            ax.set_xlim(np.min(xx), np.max(xx))
            ax.set_ylim(np.min(yy), np.max(yy))
            plt.show()
        else:
            ax.contourf(xx,yy,Z, cmap=cm)
            ax.scatter(X[cols[0]], X[cols[1]], c=y, cmap=cm_bright, \
                edgecolors='k', alpha=.7)
            ax.set_xlim(np.min(xx), np.max(xx))
            ax.set_ylim(np.min(yy), np.max(yy))
            ax.axis('off')

    else: # 1D plot (histograms)
        #cm = plt.cm.RdBu
        X_hist = [] # List of each group class values in each column
        for group in set(y):
            X_hist.append(X[cols][y==group])

        if ax==None: # Check if it is part of subplot or not
            hist_values = plt.hist(X_hist, stacked=True, alpha=.7, color=['red', 'blue'])
            xx, yy, Z = get_mesh_coordinates(clf, X, y, cols, hist_values)
            plt.contourf(xx,yy,Z, cmap=cm)
            plt.show()
        else: # in case it is part of subplot
            hist_values = ax.hist(X_hist, stacked=True, alpha=.7, color=['red', 'blue'])
            xx, yy, Z = get_mesh_coordinates(clf, X, y, cols, hist_values)
            ax.contourf(xx,yy,Z, cmap=cm)
            ax.axis('off')


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

def round_multiple(x, base=5):
    return int(base * np.round(float(x)/base))

def get_mesh_coordinates(clf, X, y, cols, hist_values=None):
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
        n_steps = 11 # the value is 11 since there's 11 values in [0,10]
        min_x, max_x = np.min(X[cols[0]]), np.max(X[cols[0]])
        min_y, max_y = np.min(X[cols[1]]), np.max(X[cols[1]])
        x = np.linspace(min_x, max_x, n_steps)
        y = np.linspace(min_y, max_y, n_steps)

        xx, yy = np.meshgrid(x, y)

        # Create matrix if features that are going to be mapped
        X_grid = create_X_grid(X, np.c_[xx.ravel(), yy.ravel()], cols)

        # Get prediction values (either probabilities or from decision function)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_grid)
        else:
            Z = clf.predict_proba(X_grid)[:, 1]

        Z = Z.reshape(xx.shape)

        return xx, yy, Z
    else:
        # 1D coordinates
        n_steps = 11 # the value is 11 since there's 11 values in [0,10]
        min_x, max_x = np.min(X[cols]), np.max(X[cols])
        y_width = round_multiple(1.3*np.max(hist_values[0]), base=5)#max_x - min_x
        x = np.linspace(min_x, max_x, n_steps)
        y = np.linspace(0, y_width, n_steps)

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







    


