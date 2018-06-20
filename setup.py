#! /usr/bin/env python
#
# Copyright (C) 2018 Fernando Marcos Wittmann

SHORT_DESCRIPTION = "VisualML: Visualization of Multi-Dimensional Machine Learning Models"

LONG_DESCRIPTION = """\
Visual ML is a library for visualizing the decision boundary of 
machine learning models from Sklearn using 2D projections of pairs
of features. Here's an example:
```
>>> import visualml as vml
>>> import pandas as pd
>>> from sklearn.datasets import make_classification
>>> from sklearn.ensemble import RandomForestClassifier as RF
 
>>> # Create a toy classification dataset
>>> feature_names = ['A','B','C','D']
>>> X, y = make_classification(n_features=4, random_state=42)
 
>>> # The visualization is only supported if X is a pandas df
>>> X = pd.DataFrame(X, columns=feature_names)
 
>>> # Train a classifier
>>> clf = RF(random_state=42).fit(X,y) 
 
>>> # Plot decision boundary grid
>>> vml.decision_boundary_grid(clf, X, y)
```

"""

DISTNAME = 'visualml'
AUTHOR = 'Fernando Marcos Wittmann'
AUTHOR_EMAIL = 'fernando.wittmann@gmail.com'
DOWNLOAD_URL = 'https://github.com/wittmannf/visual-ml/'
VERSION = '0.1b1' #.dev3' # This is going to be the only time that a beta version is released at PyPi. The following alpha, beta and dev versions are going to be released at https://test.pypi.org/project/visualml/


try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    try:
        import sklearn
    except ImportError:
        install_requires.append('sklearn')
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')

    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            "Programming Language :: Python :: 2",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        description=SHORT_DESCRIPTION,
        #entry_points={
        #    'console_scripts': [
        #        'visualml=visualml.cli:main',
        #    ],
        #},
        install_requires=install_requires,
        license="BSD-4-Clause",
        long_description=LONG_DESCRIPTION,
        include_package_data=True,
        keywords='visualml',
        name='visualml',
        packages=['visualml'],
        url=DOWNLOAD_URL,
        version=VERSION,
        zip_safe=False,
    )
