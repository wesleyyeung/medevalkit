import os
from setuptools import setup, find_packages

# Optional: dynamically read README
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='medevalkit',
    version='0.1.2',
    description='A toolkit for evaluating machine learning models for healthcare applications.',
    long_description=read("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author='Wesley Yeung',
    author_email='corona-monody-7e@icloud.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
       'lifelines',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scikit-survival',
        'seaborn'
    ],
    python_requires='>=3.7',
)