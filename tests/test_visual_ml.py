import pandas as pd
import numpy as np
import visualml as vml
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt


def test_decision_boundary_grid(n_feats=4, feat_list=None):
    columns_list = ['A','B','C','D','E', 'F', 'G', 'H', 'I']
    X, y = make_classification(n_features=n_feats, random_state=42)
    X = pd.DataFrame(X, columns=columns_list[:n_feats])
    clf = RF(random_state=42).fit(X,y)
    #fig, ax = plt.subplots(2,1)
    #vml.plot_decision_boundary(clf, X, y, 'A', ax=ax[0])
    #vml.plot_decision_boundary(clf, X, y, 'B', ax=ax[1])
    #plt.show()
    obj = vml.decision_boundary_grid(clf, X, y, feat_list=feat_list)#, figsize=(10,10))
    print(obj)


def test_plot_decision_boundary(input_dim='1D'):
    if input_dim=='1D':
        X, y = make_classification(n_features=4, random_state=42)
        clf = SVC(random_state=42).fit(X,y)
        X = pd.DataFrame(X, columns=['A','B','C','D'])
        vml.plot_decision_boundary(clf, X, y, 'A')

    elif input_dim=='1Dlist':
        X, y = make_classification(n_features=4, random_state=42)
        clf = SVC(random_state=42).fit(X,y)
        X = pd.DataFrame(X, columns=['A','B','C','D'])
        vml.plot_decision_boundary(clf, X, y, ['A'])

    elif input_dim=='2D':
        X, y = make_classification(n_features=4, random_state=42)
        clf = SVC(random_state=42).fit(X,y)
        X = pd.DataFrame(X, columns=['A','B','C','D'])
        vml.plot_decision_boundary(clf, X, y, ['A','C'])

    else:
        print("The parameter's value input_dim has to be either '1D' or '2D'")

def test_create_X_grid():
    X = pd.DataFrame(np.ones([5,5]), columns=['A','B','C','D','E'])
    x = np.ones([5,2])
    X_map = vml._create_X_grid(X, x, ['B', 'D'])
    print("Input is {}, {} and ['B', 'D']".format(X, x))
    print("Output is {}".format(X_map))

def test_get_mesh_coordinates(input_dim='1D'):
    if input_dim=='1D':
        X, y = make_classification(n_features=4, random_state=42)
        clf = SVC(random_state=42, probability=True).fit(X,y)
        X = pd.DataFrame(X, columns=['first','second','third','fourth'])

        print("Testing 1D input")
        print("input is {}, {}, {} and {}".format(clf, X, y, 'first'))
        xx, yy, Z = vml._get_mesh_coordinates(clf, X, y, 'first', [1, 2, 4])
        print("output is {}, {} and {}".format(xx, yy, Z))
        cm = plt.cm.RdBu
        plt.contourf(xx,yy,Z, cmap=cm)
        plt.show()
    elif input_dim=='2D':
        X, y = make_classification(n_features=4, random_state=42)
        clf = SVC(random_state=42, probability=True).fit(X,y)
        X = pd.DataFrame(X, columns=['A','B','C','D'])
        print("input is {}, {}, {} and {}".format(clf, X, y, ['A', 'B']))
        xx, yy, Z = vml._get_mesh_coordinates(clf, X, y, ['A', 'B'])
        print("output is {}, {} and {}".format(xx, yy, Z))
        cm = plt.cm.RdBu
        plt.contourf(xx,yy,Z, cmap=cm)
        plt.show()
    else:
        print("The parameter's value input_dim has to be either '1D' or '2D'")


def main():
#   test_create_X_grid()
#    test_get_mesh_coordinates()
#    test_plot_decision_boundary(input_dim='1D')
    test_decision_boundary_grid(n_feats=5, feat_list=['D','B','A'])
    pass

if __name__ == '__main__':
    main()