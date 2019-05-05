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

- Install tensorflow
    - $ pip install --upgrade tensorflow
    - or 
    - pip install tensorflow==2.0.0-alpha0 
    
- Verify tensorflow installation
    - v1
        - $ python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"    
## How to use the examples

The TensorFlow examples has two flavours: v1 and v2 (alpha). Following are details of the examples and corresponding program files. 
- v1
    - Basic Evaluation
        - basic-evaluation.py 
- v2 (alpha)
    - Hello World
        - Basic 
            -  basic-hello-world.py
        - Advanced 
            - advanced-hello-world.py
    
    - Classification Examples
        - Text Classification
            - text-classification.py
        - Image Classification
            - image-classification.py
        - Structured Data Classification
            - structured-data-classification.py
    
    - Exploring Datasets
        - dataset-example.py
    
    - Using Keras API
        - Keras API Basic
            - keras-basics.py
        - Keras API Overview
            - keras-overview.py
    
    - Regression Example
        - fuel-efficiency-prediction-regression.py
    
    - Tensorboard Usage
        - tensorboard-example.py


## Reference
https://www.tensorflow.org/alpha