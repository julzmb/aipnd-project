'''
 * call getargs()
 * call getmodel()
    - gets the pretrained model
    - freezes the pretraind layers
    - replaces the final layers (sometimes called the bottelneck) with our classifier layers
    - returns the mixed model (mixed because some layers are trained and some arent)
 * call train()
 * call export()
'''
def main():
    getargs()
    getdataloaders()
    getmodel()
    train()
    test()
    export()

'''
 * create & configure parser
 * call parse_args
 * return parser object
'''
def getargs():
    pass

'''
 * input: data_dir (from args)
 * input: batch size
 * output: training dataloader, validation dataloader, testing dataloader
 * calls to transforms.Compose
 * calls to datasets.ImageFolder(path, transform)
 * calls to torch.utils.data.DataLoader(dataset, batch_size=???, shuffle[training only])
'''
def getdataloaders():
    pass

'''
 * input: model name (from args)
 * input: #of hidden layers in classifier
    - Accept a list of ints
    - the length of the list indicates the number of hidden layers
    - the values of in the list indiate the width at each layer
 * input: # of classes
 * input: dropout probability
 
 * test if the given model name is present in torchvision.models
    - raise ModelNotFoundError if the provided model name isnt found
 * freeze layers
 * determine the "classifer layer" to be overwritten
 * build the new classifier
 * overwrite the pretrained classifier
 * return modified model
'''
def getmodel():
    pass

'''
 * input: model
 * input: learning rate
 * input: # of epochs
 * input: device [cpu/cuda] (derived from args)
 * input: training set dataloader
 * input: validation set dataloader
 * output: **None** ?
 * model.to(device) is called
 * for the # of epochs
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
 * input: model
 * input: test set dataloader
 * model.eval() [ 1 time ]
 * with torch.no_grad()
 * images / labels loaded from dataloader
 * images/labels .to(device)
 * mode.forward(...)
 * calculate minibatch loss
 * accuracy calculation
 * calls print_loss_and_accuracy(...)
'''
def test():
    pass

'''
 * input: save_dir (from args)
 * input: model
 * input: dataset class_to_index
 * call torch.save with dict and path
    - dict contains "class to idx" info
    - dict contains model.state_dict()
'''
def export():
    pass

if __name__ == '__main__':
    main()