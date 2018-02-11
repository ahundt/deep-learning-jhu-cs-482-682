# deep-learning-jhu-cs-482-682
Deep learning JHU CS 482 682 assignments

[![Build Status](https://travis-ci.com/ahundt/deep-learning-jhu-cs-482-682.svg?token=PLqid21E6Q2dJvLJs4aD&branch=master)](https://travis-ci.com/ahundt/deep-learning-jhu-cs-482-682)


# Programming Assignment 1

This assignment should be done in groups of 3. A minimum of 4 answers should be submitted every 7 days starting on the release date of the assignment. This assignment will take several days of CPU time if you don't have a GPU, and that's the reason for the staggered deadlines.

We will be primarily using the [Fashion-MNIST](https://arxiv.org/pdf/1708.07747.pdf) dataset, with a couple of additional runs on the original MNIST dataset.

![Fashion-MNIST](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

A single 10 epoch run should take about 15 minutes on a 2 year old laptop.


## Installation


Install pytorch as detailed at [pytorch.org](http://pytorch.org/):

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
```

If you have a GPU you'd like to use installation would be different for every machine so, unfortunately, we can only provide support for CPU considering we have such a large class.


```

# locally installed libraries
if [ -d $HOME/lib ] ; then
  # load libraries and programs installed locally
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib
  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HOME/lib
  export PYTHONPATH=$PYTHONPATH:$HOME/lib
fi

# locally installed binaries
if [ -d $HOME/bin ] ; then
  export PATH=$PATH:$HOME/bin
fi

# pip installs user packages here, for example:
# pip3 install numpy --upgrade --user
# https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUSERBASE
if [ -d $HOME/.local/bin ] ; then
  export PATH=$HOME/.local/bin:$PATH
fi

# pip installs user packages here, for example:
# pip3 install numpy --upgrade --user
if [ -d $HOME/.local ] ; then
  export PYTHONUSERBASE=$HOME/.local/
fi

```

Install the relevant libraries:

```
pip3 install tensorflow-tensorboard tensorboardX torchvision onnx tqdm --user --upgrade
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

## Questions

For each of the following situations analyze **What, How and, Why**:

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
  - Every run should

Extra credit if you set up hyperparameter tuning to run everything in one go.
There is also no need to re-run the default setting over and over again, just re-use a single default run where it is reasonable.

### Varying Datasets

1. What is the difference between the performance of mnist and fashion-mnist?
    - Take a screenshot of the images from each in tensorboard.

### Varying Hyperparameters

2. Train for twice as many epochs for both mnist and fashion_mnist.
    - [Fashion 10 epochs, MNIST 10 epochs, Fashion 20 epochs, MNIST 20 epochs]
    - How is this similar and different previous runs?

3. Change the SGD Learning Rate
    - [0.1x, 1x, +10x]

4. Compare Optimizers
    - [SGD, Adam, Rmsprop]

5. Set the dropout layer to
    - [0, 0.25, 0.5, 0.9, 1]

6. Change the batch size:
     - [1/8x, 1x, 8x]

7. Change the number of output channels in each convolution and the first Linear layer.
    - [0.5x, 1x, 2x]
    - Note: The input values of each layer will need to match the previous layer.

8. Add a Batch Normalization Layer after the first convolution.

9. Add a Dropout layer immediately after the Batch Normalization from the previous question.

10. Move the Batch Normalizaton just below the Dropout from the previous question.
    - You may want to do a quick search of the current literature for this one.

12. Add one extra Conv2D layer, and remove another layer of your choice.
    - In addition to the standard questions, what did you choose and why?


### Become the ultimate Fashion-MNIST model

13. Create the best model you can on Fashion-MNIST based on your experience from the previous questions.
    - A minimum of 90% validation accuracy is required for full credit.
    - Make sure to save your best model checkpoints or you'll be out of luck.
    - Feel free to use outside literature
    - Please write your own code
    - The best performer in the class will get a prize!

### Fine tuning

14. Evaluate your "ultimate Fashion-MNIST model" by loading the trained weights and running on MNIST without changing it at all.

15. Reduce your SGD learning rate by 100x, and train mnist on your ultimate Fashion-MNIST model
     - Compare this to your original MNIST training run