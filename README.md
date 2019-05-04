# Machine Learning Examples
This repo contains a set of machine learning examples

## Prerequisites
- Python is configured
- Virtualenv is configured

## Environment used for testing
    - OS: Ubuntu 18.04 LTS

### Instructions
- install pre-requisites
    - python3
        - on Ubuntu 18.04, python3 is installed by default
    - pip3
        - $ sudo apt install python3-pip
    - vitualenv
        - $ pip3 install virtualenv
        or
        - $ sudo apt install virtualenv (preferred)

- verify prerequisites installation 
    - $ python3 --version
    - $ pip3 --version
    - $ virtualenv --version

- create virtualenv
    - $ virtualenv --system-site-packages -p python3 ./venv
    - $ source ./venv/bin/activate
    - $ pip install --upgrade pip
    - $ pip list

- install tensorflow
    - $ pip install --upgrade tensorflow
    - or 
    - pip install tensorflow==2.0.0-alpha0 
    
- verify tensorflow installation (for v1)
    - $ python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"    

# Reference
https://www.tensorflow.org/alpha