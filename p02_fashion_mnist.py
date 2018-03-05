from __future__ import print_function
import os
import argparse
import datetime
import six
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np

from tensorboardX import SummaryWriter
import callbacks

# Training settings
parser = argparse.ArgumentParser(description='Deep Learning JHU Assignment 1 - Fashion-MNIST')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TB',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='O',
                    help='Optimizer options are sgd, p1sgd, adam, rms_prop')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MO',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='I',
                    help="""how many batches to wait before logging detailed
                            training status, 0 means never log """)
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='Options are mnist and fashion_mnist')
parser.add_argument('--data_dir', type=str, default='../data/', metavar='F',
                    help='Where to put data')
parser.add_argument('--name', type=str, default='', metavar='N',
                    help="""A name for this training run, this
                            affects the directory so use underscores and not spaces.""")
parser.add_argument('--model', type=str, default='default', metavar='M',
                    help="""Options are default, P2Q7DefaultChannelsNet,
                    P2Q7HalfChannelsNet, P2Q7DoubleChannelsNet,
                    P2Q8BatchNormNet, P2Q9DropoutNet, P2Q10DropoutBatchnormNet,
                    P2Q11ExtraConvNet, P2Q12RemoveLayerNet, and P2Q13UltimateNet.""")
parser.add_argument('--print_log', action='store_true', default=False,
                    help='prints the csv log when training is complete')

required = object()


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Add a timestamp to your training run's name.
    """
    # http://stackoverflow.com/a/5215012/99379
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

# choose the dataset


def prepareDatasetAndLogging(args):
    # choose the dataset
    if args.dataset == 'mnist':
        DatasetClass = datasets.MNIST
    elif args.dataset == 'fashion_mnist':
        DatasetClass = datasets.FashionMNIST
    else:
        raise ValueError('unknown dataset: ' + args.dataset + ' try mnist or fashion_mnist')

    training_run_name = timeStamped(args.dataset + '_' + args.name)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Create the dataset, mnist or fasion_mnist
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    training_run_dir = os.path.join(args.data_dir, training_run_name)
    train_dataset = DatasetClass(
        dataset_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataset = DatasetClass(
        dataset_dir, train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Set up visualization and progress status update code
    callback_params = {'epochs': args.epochs,
                       'samples': len(train_loader) * args.batch_size,
                       'steps': len(train_loader),
                       'metrics': {'acc': np.array([]),
                                   'loss': np.array([]),
                                   'val_acc': np.array([]),
                                   'val_loss': np.array([])}}
    if args.print_log:
        output_on_train_end = os.sys.stdout
    else:
        output_on_train_end = None

    callbacklist = callbacks.CallbackList(
        [callbacks.BaseLogger(),
         callbacks.TQDMCallback(),
         callbacks.CSVLogger(filename=training_run_dir + training_run_name + '.csv',
                             output_on_train_end=output_on_train_end)])
    callbacklist.set_params(callback_params)

    tensorboard_writer = SummaryWriter(log_dir=training_run_dir, comment=args.dataset + '_embedding_training')

    # show some image examples in tensorboard projector with inverted color
    images = 255 - test_dataset.test_data[:100].float()
    label = test_dataset.test_labels[:100]
    features = images.view(100, 784)
    tensorboard_writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
    return tensorboard_writer, callbacklist, train_loader, test_loader


# TODO Add classes for every option listed under the --model parser argument above.

# Define the neural network classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # F is just a functional wrapper for modules from the nn package
        # see http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class P2Q7HalfChannelsNet(nn.Module):
    def __init__(self):
        super(P2Q7HalfChannelsNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q7DoubleChannelsNet(nn.Module):
    def __init__(self):
        super(P2Q7DoubleChannelsNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q8BatchNormNet(nn.Module):
    def __init__(self):
        super(P2Q8BatchNormNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q9DropoutNet(nn.Module):
    def __init__(self):
        super(P2Q9DropoutNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q10DropoutBatchnormNet(nn.Module):
    def __init__(self):
        super(P2Q10DropoutBatchnormNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q11ExtraConvNet(nn.Module):
    def __init__(self):
        super(P2Q11ExtraConvNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q12RemoveLayerNet(nn.Module):
    def __init__(self):
        super(P2Q12RemoveLayerNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


class P2Q13UltimateNet(nn.Module):
    def __init__(self):
        super(P2Q13UltimateNet, self).__init__()
        # TODO Implement me
        raise NotImplementedError

    def forward(self, x):
        # TODO Implement me
        raise NotImplementedError


def chooseModel(model_name='default', cuda=True):
    # TODO add all the other models here if their parameter is specified
    if model_name == 'default' or model_name == 'P2Q7DefaultChannelsNet':
        model = Net()
    elif model_name in globals():
        model = globals()[model_name]()
    else:
        raise ValueError('Unknown model type: ' + model_name)

    if args.cuda:
        model.cuda()
    return model


def chooseOptimizer(model, optimizer='sgd'):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError('Unsupported optimizer: ' + args.optimizer)
    return optimizer


def train(model, optimizer, train_loader, tensorboard_writer, callbacklist, epoch, total_minibatch_count):
    # Training
    model.train()
    correct_count = np.array(0)
    for batch_idx, (data, target) in enumerate(train_loader):
        callbacklist.on_batch_begin(batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.nll_loss(output, target)

        # Backpropagation step
        loss.backward()
        optimizer.step()

        # The batch has ended, determine the
        # accuracy of the predicted outputs
        _, argmax = torch.max(output, 1)

        # target labels and predictions are
        # categorical values from 0 to 9.
        accuracy = (target == argmax.squeeze()).float().mean()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_count += pred.eq(target.data.view_as(pred)).cpu().sum()

        batch_logs = {
            'loss': np.array(loss.data[0]),
            'acc': np.array(accuracy.data[0]),
            'size': np.array(len(target))
        }

        batch_logs['batch'] = np.array(batch_idx)
        callbacklist.on_batch_end(batch_idx, batch_logs)

        if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:
            # put all the logs in tensorboard
            for name, value in six.iteritems(batch_logs):
                tensorboard_writer.add_scalar(name, value, global_step=total_minibatch_count)

            # put all the parameters in tensorboard histograms
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                tensorboard_writer.add_histogram(name, param.data.cpu().numpy(), global_step=total_minibatch_count)
                tensorboard_writer.add_histogram(name + '/gradient', param.grad.data.cpu().numpy(), global_step=total_minibatch_count)

        total_minibatch_count += 1

    # display the last batch of images in tensorboard
    img = torchvision.utils.make_grid(255 - data.data, normalize=True, scale_each=True)
    tensorboard_writer.add_image('images', img, global_step=total_minibatch_count)

    return total_minibatch_count


def test(model, test_loader, tensorboard_writer, callbacklist, epoch, total_minibatch_count):
    # Validation Testing
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_loader, desc='Validation')
    for data, target in progress_bar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_size = np.array(len(test_loader.dataset), np.float32)
    test_loss /= test_size

    acc = np.array(correct, np.float32) / test_size
    epoch_logs = {'val_loss': np.array(test_loss),
                  'val_acc': np.array(acc)}
    for name, value in six.iteritems(epoch_logs):
        tensorboard_writer.add_scalar(name, value, global_step=total_minibatch_count)
    callbacklist.on_epoch_end(epoch, epoch_logs)
    progress_bar.write(
        'Epoch: {} - validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc


def run_experiment(args):
    total_minibatch_count = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    epochs_to_run = args.epochs
    tensorboard_writer, callbacklist, train_loader, test_loader = prepareDatasetAndLogging(args)
    model = chooseModel(args.model)
    # tensorboard_writer.add_graph(model, images[:2])
    optimizer = chooseOptimizer(model, args.optimizer)
    # Run the primary training loop, starting with validation accuracy of 0
    val_acc = 0
    callbacklist.on_train_begin()
    for epoch in range(1, epochs_to_run + 1):
        callbacklist.on_epoch_begin(epoch)
        # train for 1 epoch
        total_minibatch_count = train(model, optimizer, train_loader, tensorboard_writer,
                                      callbacklist, epoch, total_minibatch_count)
        # validate progress on test dataset
        val_acc = test(model, test_loader, tensorboard_writer,
                       callbacklist, epoch, total_minibatch_count)
    callbacklist.on_train_end()
    tensorboard_writer.close()

    if args.dataset == 'fashion_mnist' and val_acc > 0.92 and val_acc <= 1.0:
        print("Congratulations, you beat the Question 13 minimum of 92 with ({:.2f}%) validation accuracy!".format(val_acc))

if __name__ == '__main__':
    args = parser.parse_args()
    # Run the primary training and validation loop
    run_experiment(args)

