# deep-learning-jhu-cs-482-682
Deep learning JHU CS 482 682 assignments

[![Build Status](https://travis-ci.com/ahundt/deep-learning-jhu-cs-482-682.svg?token=PLqid21E6Q2dJvLJs4aD&branch=master)](https://travis-ci.com/ahundt/deep-learning-jhu-cs-482-682)


# Programming Assignment 1 (100 points total)

This assignment should be done in groups of 3. A minimum of 4 answers should be submitted every 7 days starting on the release date of the assignment. This assignment will take several days of CPU time if you don't have a GPU, and that's the reason for the staggered deadlines.

We will be primarily using the [Fashion-MNIST](https://arxiv.org/pdf/1708.07747.pdf) dataset, with a couple of additional runs on the original MNIST dataset.

![Fashion-MNIST](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

A CPU only 10 epoch run should take about 15 minutes on a 4 year old laptop. Much of your time might be executing training runs in the background on your computer. This means you need to plan for up to 24 hours of total training time! Making a plan for this assignment, installing the software ahead of time, setting a schedule, and submitting on time is 100% your responsibility.

## Installation

Install [miniconda](https://conda.io/docs/user-guide/install/index.html), a python package manager.


Install [pytorch](http://pytorch.org/):

```
conda create -q -n dl-jhu-env python=3.6 pip numpy chainer torchvision tensorflow-tensorboard tqdm pytorch-cpu torchvision -c pytorch
```

If you have a GPU you'd like to use installation would be different for every machine so, unfortunately, we can only provide support for CPU considering we have such a large class.


Install the visualization tools:

```
source activate dl-jhu-env
which python
conda list
pip install --upgrade pytest flake8 tensorboardX onnx
```

Fork and then clone the repository on GitHub

```
git clone http://path/to/repo
cd repo
```

Make sure the folder `/path/to/repo/../data` aka `/path/to/data` is available, or set `--data-dir` when running commands below.

### Steps to run


Train the provided model on mnist

```
python2 a1_mnist_fashion.py --dataset fashion_mnist --epochs 10
```

Train the default model on Fashion-MNIST


```
python2 a1_mnist_fashion.py --dataset fashion_mnist --epochs 10
```

Look at the results on tensorboard:

```
tensorboard --port 8888 --logdir ../data
```

Open your web browser and go to [http://localhost:8888](http://localhost:8888).


Run all your experiments:

```
sh 1_fashion_mnist_experiments.sh
```


## Questions

For each of the following questions (#1-15) analyze **What, How and, Why**:

  - What does each change do mathematically?
  - What does each change do algorithmically?
  - How and why does the loss, accuracy, validation loss, and validation accuracy change?
  - How and why does the training time change? (if at all)
  - Explain why you would want to apply such a change to your model.
  - After each question return to the parameters to their original settings unless the next question says otherwise.
  - Run on Fashion-MNIST unless the instructions say otherwise.
  - Include a screenshot of your tensorboard scalars for each situation and compare the effect of the hyperparameter changes.
      - The labels and pictures must be very clear.
      - Don't forget you can move the folders from your `../data` directory and back into it.

Extra credit if you set up hyperparameter tuning to run everything in one go.
There is also no need to re-run the default setting over and over again, just re-use a single default run where it is reasonable.

### Varying Datasets (3 points)

1. Compare the performance of mnist and fashion-mnist

### Varying Hyperparameters (33 points)

2. Train for twice as many epochs for both mnist and fashion_mnist.
    - [Fashion 10 epochs, MNIST 10 epochs, Fashion 20 epochs, MNIST 20 epochs]
    - How is this similar and different previous runs?

3. Change the SGD Learning Rate by a factor of
    - [0.1x, 1x, 10x]

4. Compare Optimizers
    - [SGD, Adam, Rmsprop]

5. Set the dropout layer to a dropout rate of
    - [0, 0.25, 0.5, 0.9, 1]

6. Change the batch size:
     - [1/8x, 1x, 8x]

7. Change the number of output channels in each convolution and the first Linear layer.
    - [0.5x, 1x, 2x]
    - Note: The input values of each layer will need to match the previous layer.

8. Add a Batch Normalization Layer after the first convolution.

9. Add a Dropout layer immediately after the Batch Normalization from the previous question.

10. Move the Batch Normalizaton layer just below the Dropout layer from the previous question.
    - Compare 9 with 10 and explain what happened.
    - You may want to do a quick search of the current literature for this one.

11. Add one extra Conv2D layer

12. Remove a layer of your choice
    - In addition to the standard questions, what did you choose and why?


### Become the ultimate Fashion-MNIST model (50 points)

13. Create the best model you can on Fashion-MNIST based on your experience from the previous questions.
    - A minimum of 92% validation accuracy is required for full credit.
    - Make sure to save your best model checkpoints or you'll be out of luck.
    - Feel free to use outside literature
    - Please write your own code
    - The best performer in the class will get a prize!

### Fine tuning between datasets (14 points)

14. Evaluate your "ultimate Fashion-MNIST model" by loading the trained weights and running on MNIST without changing the Fashion-MNIST weights at all.

15. Reduce your SGD learning rate by 100x, and train MNIST on your ultimate Fashion-MNIST model
     - Compare this to your original MNIST training run and the previous question


## Requirements

- Address all TODOs including those in
    - `1_fashion_mnist.py`
    - `1_fashion_mnist_experiments.sh`
    - `1_fashion_mnis_answers.md`
- Your code must pass [pep8](https://www.python.org/dev/peps/pep-0008/) style checks.
- Your code must pass the travis CI tests, and that is where we will evaluate your ultimate model's results.
- You must provide a markdown file `1_fashion_mnist_answers.md` with your answers to questions 1-15
    - It should include the accompanying tensorboard photos.
- The line with `python 1_fashion_mnist.py --dataset fashion_mnist` is the only line in `.travis.yml` which you may modify to improve your ultimate model's results.
- If execution of the travis script does not end indicating you pass, question 13 is considered incomplete.
- It should be easy to view a diff including the changes you made for your final code submission.
- You may be required to merge a change to this assignment if a correction is required.
- Have fun!
