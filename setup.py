#! /usr/bin/env python
#
# Copyright (C) 2018 Fernando Marcos Wittmann

DESCRIPTION = "Visual ML: visualization of machine learning models"
LONG_DESCRIPTION = """\
Visual ML is a library for visualizing the decision boundary of 
machine learning models from sklearn using 2D projections of pairs
of features. 
"""

DISTNAME = ''
MAINTAINER = 'Fernando Marcos Wittmann'
MAINTAINER_EMAIL = 'fernando.wittmann@gmail.com'
DOWNLOAD_URL = 'https://github.com/wittmannf/visual_ml/'
VERSION = '0.1'

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

    setup(name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=install_requires,
        packages=['visual_ml', 'visual_ml.tests'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering :: Visualization',
                     'Topic :: Multimedia :: Graphics',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
          )