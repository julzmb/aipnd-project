import argparse
import torch
import torchvision

_DATALOADER_NUM_WORKERS = 8

'''
call getargs()
call getmodel()
    - gets the pretrained model
    - freezes the pretraind layers
    - replaces the final layers (sometimes called the bottelneck) with our classifier layers
    - returns the mixed model (mixed because some layers are trained and some arent)
call train()
call export()
'''
def main():
    getargs() # args = getargs().parse_args()
    getdataloaders('', 1) #  using dummy args
    getmodel()
    train()
    test()
    export()

'''
create & configure parser
return parser object
parser to be called externally
'''
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', required=True)
    choices = [
        'alexnet',
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'googlenet',
        'inception_v3',
        'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
        'mobilenet_v2',
        'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
        'resnext101_32x8d', 'resnext50_32x4d',
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
        'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        'squeezenet1_0', 'squeezenet1_1',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
        'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
        'wide_resnet101_2', 'wide_resnet50_2'
    ]
    parser.add_argument('--arch', required=True, choices=choices)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--hidden_units', type=int, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')

    '''TODO:
    - dropout drop probability
    - data augmentation params(random rotation)
    '''
    return parser

'''
input: data_dir (from args)
input: batch size
output: training dataloader, validation dataloader, testing dataloader
calls to transforms.Compose
calls to datasets.ImageFolder(path, transform)
calls to torch.utils.data.DataLoader(dataset, batch_size=???, shuffle[training only])
'''
def getdataloaders(datadir, batch_size):
    train_ds = torchvision.datasets.ImageFolder(
        datadir + '/train',
        transform=torchvision.transforms.Compose([])
    )  # dummy root value
    val_ds = torchvision.datasets.ImageFolder(
        datadir + '/valid',
        transform=torchvision.transforms.Compose([])
    )  # dummy root value
    test_ds = torchvision.datasets.ImageFolder(
        datadir + '/test',
        transform=torchvision.transforms.Compose([])
    )  # dummy root value
    return (
        torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=_DATALOADER_NUM_WORKERS,
            shuffle=True
        ),
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=_DATALOADER_NUM_WORKERS
        ),
        torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=_DATALOADER_NUM_WORKERS
        )
    )

'''
input: model name (from args)
input: #of hidden layers in classifier
    - Accept a list of ints
    - the length of the list indicates the number of hidden layers
    - the values of in the list indiate the width at each layer
input: # of classes
input: dropout probability

test if the given model name is present in torchvision.models
    - raise ModelNotFoundError if the provided model name isnt found
freeze layers
determine the "classifer layer" to be overwritten
build the new classifier
overwrite the pretrained classifier
return modified model
'''
def getmodel():
    pass

'''
input: model
input: learning rate
input: # of epochs
input: device [cpu/cuda] (derived from args)
input: training set dataloader
input: validation set dataloader
output: **None** ?
model.to(device) is called
for the # of epochs
    * train_loop
        - model.train() [ 1 time ]
        - images/ labels from dataloader
        - images / labels .to(device)
        - optimizer.zero_grad()
        - logps = model.forward(images)
        - loss = criterion(logps, labels)
        - loss.backward()
        - optimizer.step()
        - ***accumulates loss.item()
        - calls print_training_loss
    * validation_loop
        - model.eval() [ 1 time ]
        - with torch.no_grad()
        - images/labels from dataloader
        - images/labels .to(device)
        - logps = model.forward(images)
        - calculate minibatch loss
        - accuracy calculation
            - get probs, get top1 estimate, derive accuracy
        - calls print_loss_and_accuracy(...)
'''
def train():
    pass

'''
input: model
input: test set dataloader
model.eval() [ 1 time ]
with torch.no_grad()
images / labels loaded from dataloader
images/labels .to(device)
mode.forward(...)
calculate minibatch loss
accuracy calculation
calls print_loss_and_accuracy(...)
'''
def test():
    pass

'''
input: save_dir (from args)
input: model
input: dataset class_to_index
call torch.save with dict and path
    - dict contains "class to idx" info
    - dict contains model.state_dict()
'''
def export():
    pass

if __name__ == '__main__':
    main()