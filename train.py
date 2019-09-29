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
    getmodel()
    train()
    test()
    export()

'''
 * configure parser
 * call parse_args
 * return args object
'''
def getargs():
    pass

'''
 * input: model name
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

def train():
    pass

def test():
    pass

def export():
    pass

if __name__ == '__main__':
    main()