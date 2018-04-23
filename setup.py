#! /usr/bin/env python
#
# Copyright (C) 2018 Fernando Marcos Wittmann

LONG_DESCRIPTION = """\
Visual ML is a library for visualizing the decision boundary of 
machine learning models from sklearn using 2D projections of pairs
of features. 
"""

DISTNAME = 'visualml'
AUTHOR = 'Fernando Marcos Wittmann'
AUTHOR_EMAIL = 'fernando.wittmann@gmail.com'
DOWNLOAD_URL = 'https://github.com/wittmannf/visual_ml/'
VERSION = '0.1.0'

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
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            "Programming Language :: Python :: 2",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        description="VisualML: Visualization of multi-dimensional Machine Learning models",
        entry_points={
            'console_scripts': [
                'visualml=visualml.cli:main',
            ],
        },
        install_requires=install_requires,
        license="MIT license",
        long_description=LONG_DESCRIPTION,
        include_package_data=True,
        keywords='visualml',
        name='visualml',
        packages=['visualml'],
        url='https://github.com/wittmannf/visualml',
        version=VERSION,
        zip_safe=False,
    )
