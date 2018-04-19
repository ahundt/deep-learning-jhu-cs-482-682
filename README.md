# deep-learning-jhu-cs-482-682
Deep learning JHU CS 482 682 assignments

Build Status of [github.com/ahundt/deep-learning-jhu-cs-482-682](https://travis-ci.org/ahundt/deep-learning-jhu-cs-482-682):

[![Build Status of github.com/ahundt/deep-learning-jhu-cs-482-682](https://travis-ci.org/ahundt/deep-learning-jhu-cs-482-682.svg?branch=master)](https://travis-ci.org/ahundt/deep-learning-jhu-cs-482-682)

Public Course Starter Repository:


## Requirements for assignments

These requirements will apply to all the assignments, but we use p02 as an example:

- Answer all of the questions by filling out the markdown file `02_fashion_mnist_answers.md` with your answers
    - It should include the accompanying screenshots, if applicable.
- Address all TODOs, for example in assignment 02:
    - `p02_fashion_mnist.py`
    - `p02_fashion_mnist_experiments.sh`
    - `p02_fashion_mnist_answers.md`
    - `p02_fashion_mnist_tests.md`
- Your code must pass the travis CI tests
    - The travis CI output is where we will evaluate your ultimate model's validation performance
    - Your code must pass [pep8](https://www.python.org/dev/peps/pep-0008/) style checks  built into the travis CI tests
    - If tests do not pass, or the test raises the python `NotImplementedError` exception the question is considered incomplete.
    - Tests passing does not guarantee your code is correct.

- It should be easy to view a diff including the changes you made for your final code submission.
- You may be required to merge a change to this assignment if a correction is required.
- If you make/fix something cool that you can share with the class we'd like a pull request! (no answers please)
- We will grade the final github commit made before the deadline which passes Travis CI.
    - Travis may run beyond the deadline.
    - Out of time errors mean the test did not pass Travis CI.
- Have fun!

# GitHub and Travis-CI

## Learning to use GitHub

You may not be familiar with [GitHub](github.com), fortunately they provide many [GitHub Guides](https://guides.github.com/) to help you get started! Here are the most important:

- [Hello World](https://guides.github.com/activities/hello-world/) - the very basics
- [Mastering Markdown](https://guides.github.com/features/mastering-markdown/) - this is the format your answers file should use
- [Forking Projects](https://guides.github.com/activities/forking/) - How to work on your own personal copy
- [Setting Up GitHub Classroom Group Assignments](https://youtu.be/-52quDR2QSc) - How to set up your group
- [Merging project changes from other repositories into your fork](https://help.github.com/articles/merging-an-upstream-repository-into-your-fork/) - How to bring updates into your own personal copy of the code.

## Continuous Integration and Travis-CI

[Continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) is a way to avoid mistakes in your code by automating testing, generation of documentation, etc. every time you make a change. [Travis-CI](https://en.wikipedia.org/wiki/Travis_CI) is a company which provides computing resources and github functionality which is utilized for continuous integration in many open source projects and enterprises. [Travis-CI for beginners](https://docs.travis-ci.com/user/for-beginners/) has more details.

### Modifying your .travis.yml

If you want to run your training scripts on travis you can simply update the appropriate line.
```
python p02_fashion_mnist.py --dataset fashion_mnist
```

might become

```
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.1
```

Be sure that your unit tests are correct and passing in the final submission or it could affect your grade!

### Build Status Marker

The build status marker ([![Build Status of github.com/ahundt/deep-learning-jhu-cs-482-682](https://travis-ci.org/ahundt/deep-learning-jhu-cs-482-682.svg?branch=master)](https://travis-ci.org/ahundt/deep-learning-jhu-cs-482-682)) is set to the main assignment repository by default and does not track any other clones or forks. If you would like the marker to point to your own repository, edit this README.md file according to the [instructions in the travis-ci documentation for Embedding Status Images](https://docs.travis-ci.com/user/status-images/).

### Auto Cancellation

To speed up your build time please enable [the Travis-CI auto-cancellation feature](https://blog.travis-ci.com/2017-03-22-introducing-auto-cancellation).

## Educational membership with GitHub and Travis-CI

You may also want to sign up for the github and travis ci education programs at:

https://education.github.com/

https://education.travis-ci.com/

It will get you full access to free private repositories additional computing resources as if you were a paying developer.

# Programming Assignment 2 (100 points total)

This assignment should be done in groups of 3. A minimum of 4 answers should be submitted every 7 days starting on the release date of the assignment. This assignment will take several days of CPU time if you don't have a GPU, and that's the reason for the staggered deadlines.

We will be primarily using the [Fashion-MNIST](https://arxiv.org/pdf/1708.07747.pdf) dataset, with a couple of additional runs on the original MNIST dataset.

![Fashion-MNIST](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

A CPU only 10 epoch run should take about 5 minutes on a 4 year old laptop. Much of your time might be executing training runs in the background on your computer. This means you need to plan for the total training time! Making a plan for this assignment, installing the software ahead of time, setting a schedule, and submitting on time is 100% your responsibility.

## PyTorch and Package Installation

Install [miniconda](https://conda.io/docs/user-guide/install/index.html), a python package manager.

Make sure miniconda is on your path, you may want to add the following to your `~/.bashrc` file:

```
export PATH=~/miniconda3/bin:$PATH
```

Install [pytorch](http://pytorch.org/), you can see pytorch website for more detailed instructions:

```
conda create -q -n dl-jhu-env python=3.6 pip numpy chainer torchvision tqdm
source activate dl-jhu-env
conda install pytorch-cpu torchvision -c pytorch
```

If you have a GPU you'd like to use installation would be different for every machine so, unfortunately, we can only provide support for CPU considering we have such a large class.


Install the visualization tools:

```bash
which python # note that your python has changed
conda list
pip install --upgrade pytest flake8 tensorflow-tensorboard tensorboardX onnx
```

Make sure everything is installed correctly

```

python -c "import torch, torchvision, tensorboardX, tqdm; print('success')"

```

You should see the following if your setup is done correctly:

```
> python -c "import torch, torchvision, tensorboardX, tqdm; print('success')"
success
```

Fork and then clone the repository on GitHub

```
git clone git@github.com:path-to-your/group-repository.git
cd repo
```

Make sure the folder `/path/to/repo/../data` aka `/path/to/data` is available, or set `--data-dir` when running commands below.

### Troubleshooting the Installation

If you have trouble with tqdm, try the following install command:

```
conda install -c conda-forge tqdm
```

Tensorboard should not require tensorflow, but if you run into an error and just want to move on try the following:

```
conda install -c anaconda tensorflow tensorflow-tensorboard
```

Can't run tensorboard because the command is not found? Make sure it is on your PATH.
```
ls ~/anaconda3/bin

# Is tensorboard there?
# If so, add it to your path.

export PATH=~/anaconda3/bin:$PATH

# Also consider adding the above line to your ~/.bashrc
```


If you are on mac and you encounter protobuf errors, make sure you have [homebrew](https://brew.sh) and run:

```
brew install protobuf
```

#### JHU Ugrad machine specific errors

If you see something like:
```
conda: Command not found
```

The ugrad machines seem to default to `tcsh`, so the install steps might not work. To check your shell run:

```
echo $SHELL
```
If the file path it prints doesn't have `bash` (an incorect shell example is `tcsh`), then simply run the following to start bash:
```
bash
```

### Steps to run


Train the provided model on mnist

```
python p02_fashion_mnist.py --dataset mnist --epochs 10
```

Train the default model on Fashion-MNIST


```
python p02_fashion_mnist.py --dataset fashion_mnist --epochs 10
```

Look at the results on tensorboard:

```
tensorboard --port 8888 --logdir ../data
```

While TensorBoard is running, open your web browser and go to [http://localhost:8888](http://localhost:8888).


Run all your experiments:

```
sh p02_fashion_mnist_experiments.sh
```

Run the unit tests:

```
py.test p02_fashion_mnist_tests.py
```


## Questions

For each of the following questions (#1-8):

  - Run on Fashion-MNIST unless the instructions say otherwise.
  - Include a screenshot of your tensorboard scalars for each situation.
  - Give a very brief explanation of effect of this hyperparameter change
      - This requires that you understand and be able to articulate the difference between the various metrics and losses, as well as the conclusions you can reach based on them.
  - The labels and screenshots must be very clear, the `--name extra_description` parameter can help
  - Don't forget you can move the folders from your `../data` directory and back into it.
  - After each question, return to the parameters to their original settings unless the next question says otherwise.
  - Make sure every configuration you run with is in `p02_fashion_mnist_experiments.sh`.

There is also no need to re-run the default Net model setting over and over again, just re-use a single default run where it is reasonable.

### Varying Datasets (3 points)

1. Compare the performance of mnist and fashion-mnist

### Varying Hyperparameters (3 points each)

2. Train for twice as many epochs for both mnist and fashion_mnist.
    - [Fashion 10 epochs, MNIST 10 epochs, Fashion 20 epochs, MNIST 20 epochs]
    - How is this similar and different previous runs?

3. Change the SGD Learning Rate by a factor of
    - [0.1x, 1x, 10x]

4. Compare Optimizers
    - [SGD, Adam, Rmsprop]

5. Set the dropout layer to a dropout rate of
    - [0, 0.25, 0.5, 0.9, 1]

6. Change the batch size by a factor of:
     - [1/8x, 1x, 8x]

7. Change the number of output channels in each convolution and the first Linear layer.
    - [0.5x, 1x, 2x]
    - Note: The input values of each layer will need to match the previous layer.
    - You'll need to implement `P2Q7HalfChannelsNet` and `P2Q7DoubleChannelsNet`.

8. Add a Batch Normalization Layer after the first convolution.

9. Add a Dropout layer immediately after the Batch Normalization from the previous question.

10. Move the Batch Normalizaton layer just below the Dropout layer from the previous question.
    - Compare 9 with 10 and explain what happened.
    - You may want to do a quick search of the current literature for this one.

11. Add one extra Conv2D layer, starting from the default Net model

12. Remove a layer of your choice, starting from the default Net model
    - In addition to the standard questions, what did you choose and why?


### Become the ultimate Fashion-MNIST model (25 points)

13. Create the best model you can on Fashion-MNIST based on your experience from the previous questions.
    - A minimum of 92% validation accuracy is required for full credit.
    - Make sure to save your best model checkpoints or you'll be out of luck.
    - Feel free to use outside literature
    - Please write your own code
    - Also answer the following questions
        1. What does each change do mathematically?
        2. What does each change do algorithmically?
        3. How and why does the loss, accuracy, validation loss, and validation accuracy change?
        4. How and why does the training time change? (if at all)
        5. Explain why you would want to apply such a change to your model.
    - The best performer in the class will get a prize!

### Fine tuning between datasets (3 points each)

14. Evaluate your "ultimate Fashion-MNIST model" by loading the trained weights and running on MNIST without changing the Fashion-MNIST weights at all.

15. Reduce your SGD learning rate by 20x, and train MNIST on your ultimate Fashion-MNIST model
     - Compare this to your original MNIST training run and the previous question


# P03 Implementing Layers

Implement each of the following layers, some in functional format and others as classes. What to do:

 - You will be changing all files that begin with `p03_*`.
 - Implement every class and function with `raise NotImplementedError`.
     - The starter code provides documentation of how it should work.
     - You must manually implement the math, so you can utilize pytorch but you can't utilize an implementation from pytorch or elsewhere.
 - Backpropagation
     - For each layer that defines `backward()` you must manually implement the backpropagation step, and [pytorch custom backwards() examples can be found here](https://github.com/jcjohnson/pytorch-examples/blob/master/README.md).
     - For all others you may use the autograd functionality supplied in pytorch.
 - Unit Tests
     - For each class and function you must also implement a unit test that validates the correctness of your work.
     - Each unit test must be run in travis, and travis must pass.
     - See p02_fashion_mnist_tests.py and the pytorch source code for examples of how to write a proper unit test, some initial tests of P3SGD are already there for you.
     - All unit tests should take no longer than 120 seconds to run in total, this also means travis-ci runs should not exceed about 15 minutes.
     - The "correct" unit test answer will not be fully defined in advance. Part of this exercise is to think abut how your algorithm might fail, and come up with tests to check for those mistakes.

## Questions

1. P3SGD class (10 points)
    - Implement Stochastic Gradient Descent.
    - Nesterov momentum is optional for the hypothetical users of your class, but a requirement for the assignment.
    - Unit tests are already integrated for SGD to get you started.
2. P3Dropout class (10 points)
    - Randomly zero elements of the input tensor.
3. P3Dropout2d class (10 points)
    - Randomly zero whole channels of the input tensor.
4. P3Linear class plus a backward pass (10 points)
    - The basic `Linear` layer, sometimes also known as a dense layer or a fully connected layer.
    - You must manually implement the backwards pass.
5. p3relu function (10 points)
    - Standard Rectified Linear Unit implemented as a function.
6. P3ReLU class (10 points)
    - Standard Rectified Linear Unit implemented as a class.
7. P3ELU class plus a backward pass (10 points)
    - [Exponential Linear Units](https://arxiv.org/abs/1511.07289)
    - [ELU appears to be excellent at classifying Hot Dogs vs Not Hot Dog](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)
    - You must manually implement the backwards pass.
8. P3BCELoss (10 points)
    - [Binary Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression)
9. Create a Net (20 points)
    - Update the `Net` class to utilize each of the classes and functions you implemented.
    - Demonstrate that you are able to train the new Net to get 80% accuracy on Fashion-MNIST.
    - You must include the chart and a description in `p03_layers_answers.md` as you did in p02.
    - Run once with a 10 class Fashion-MNIST loss, and once classifying `handbag` vs `not handbag` to demonstrate `P3BCELoss`. 

6 points of each question will be devoted to the correctness of your answer and 4 points will be devoted to the quality with which your unit tests validate the correctness of your implementation. The remaining 10 points on item 10 will be based on your results in `p03_layers_answers.md`.
