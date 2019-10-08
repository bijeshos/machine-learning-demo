# Machine Learning Examples

This repo contains a set of machine learning examples using TensorFlow. (Examples using other frameworks will be added later.) 

## Prerequisites
It is assumed that the following is already configured in the local machine:
- Python
- Virtualenv

## Environment used for development/testing
- OS: Ubuntu 18.04 LTS

## Instructions
- Install pre-requisites
    - python3
        - On Ubuntu 18.04, python3 is installed by default
    - pip3
        - $ sudo apt install python3-pip
    - vitualenv
        - $ pip3 install virtualenv
        or
        - $ sudo apt install virtualenv (preferred)

- Verify prerequisites installation by running the following commands:
    - $ python3 --version
    - $ pip3 --version
    - $ virtualenv --version

- Create virtualenv
    - $ virtualenv --system-site-packages -p python3 ./venv
    - $ source ./venv/bin/activate
    - $ pip install --upgrade pip
    - $ pip list

- Install Tensorflow
    - $ pip install --upgrade tensorflow
    - or 
    - pip install tensorflow==2.0.0-beta 
    
## How to use the examples

The TensorFlow examples in this repo have two flavours: v2(beta) and v1(stable). 
Following are details of the examples and corresponding program files. 


- Basics
    - [basic-hello-world.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowbasics/basic-hello-world.py)
    - [advanced-hello-world.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowbasics/advanced-hello-world.py)
    - [temperature_conversion.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowbasics/temperature_conversion.py)

- Classification Examples
    - Text Classification
        - [text-classification.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowclassification/text-classification.py)
    - Image Classification
        - [image-classification.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowclassification/image-classification.py)
    - Structured Data Classification
        - [structured-data-classification.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowclassification/structured-data-classification.py)

- Exploring Datasets
    - [dataset-example.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowdatasets/dataset-example.py)

- Using Keras API
    - Keras API Basic
        - [keras-basics.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowkeras/keras-basics.py)
    - Keras API Overview
        - [keras-overview.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowkeras/keras-overview.py)

- Regression Example
    - [fuel-efficiency-prediction-regression.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowregression/fuel-efficiency-prediction-regression.py)

- Tensorboard Usage
    - [tensorboard-example.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflowtensorboard/tensorboard-example.py)

- v1
    - Basic Evaluation
        - [basic-evaluation.py](https://github.com/bijeshos/machine-learning-demo/blob/master/tensorflow/v1/basic-evaluation.py)

## What's next
Above mentioned are just a starting point (Thanks to TensorFlow documentation). More examples are being created. As and when they are ready, it will be added to this repository. 


## Reference
https://www.tensorflow.org/beta