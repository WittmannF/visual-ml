from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division

def plot_decision_boundaries(clf, feature_A, feature_B, target, ax=None, plt_title=''):
    # Get limits of the mesh grid
    x_min, x_max = min(feature_A), max(feature_A)
    y_min, y_max = min(feature_B), max(feature_B)
    
    # Get step of the mesh grid
    h_x = (x_max - x_min)/10
    h_y = (y_max - y_min)/10
    
    # Create the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max+h_x, h_x),
                         np.arange(y_min, y_max+h_x, h_y))
    
    # Get prediction values (either probabilities or from decision function)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    # Convert Z to 2D
    Z = Z.reshape(xx.shape)
    
    # Plot contour function    
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    if ax is None:
        ax=plt.gca()
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.8)
    ax.scatter(feature_A, feature_B, c=target, cmap=cm_bright, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.title(plt_title)
    #plt.show()
