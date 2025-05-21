import os
from setuptools import setup, find_packages

# Optional: dynamically read README
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='evalkit',
    version='0.1.0',
    description='A toolkit for evaluating machine learning models.',
    long_description=read("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author='wy',
    author_email='author_email',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'xgboost'
    ],
    python_requires='>=3.7',
)