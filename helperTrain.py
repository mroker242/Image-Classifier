import torch

def model_call(arch='vgg16'):
    '''
    Two options to choose: 'vgg16' and 'vgg13'
    '''
    from torchvision import transforms, datasets, models
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        return model
    else:
        model = models.vgg13(pretrained=True)
        return model
    
    
def load_model(model, nn, OrderedDict, n_in, n_h, n_out):
    '''
    This function does the following
    1. Freezes the borrowed parameters that does not need to be trained.
    2. Defines the classifier that will be attached to the model.
    3. Attach classifier to the model.
    '''
    ## first thing's first: freeze the parameters that are already in the model
    for param in model.parameters():
        param.requires_grad = False
    
    
    
    ## define our classifier which will attach to the model
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(n_in, n_h[0])),   
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(n_h[0],n_h[1])),
                                ('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(n_h[1], n_out)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    model.classifier = classifier
    
    
def validation(model, valid_loader, criterion):
    '''
    This validation function will be used to test accuracy as the model is being trained
    '''
    valid_loss = 0
    valid_accuracy = 0
    for images, labels in valid_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss/len(valid_loader), valid_accuracy/len(valid_loader)


def train_model(model, learning_rate, epochs, nn, optim, data, validation_data, device='cuda' ): 
    import torch
    ## this cell will actually train the classifier after being built
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # Only train the classifier parameters, feature parameters are frozen
    # epoch is changed to 1 just for testing, should be set to 7
    #epochs = epochs
    print_every = 40
    steps = 0

    # change to cuda for GPU processing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(data):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward and back passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                #SECTION FOR VALIDATION
                model.eval() # this is so that dropout is not on
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validation_data, criterion)


                print("Epoch: {}/{}... ".format(e+1, epochs),
                     "Training Loss: {:.4f}".format(running_loss/print_every),
                     "Test Loss: {:.3f}".format(test_loss),
                     "Test Accuracy: {:.3f}".format(accuracy))



                running_loss = 0

                model.train() # this is so that the dropout is turned back on
                
                
# Test accuracy on the validation data

def calculate_acc(model, data):
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            #obtain outputs from the model
            outputs = model.forward(inputs)
            
            #get the probabilities
            _, predicted = outputs.max(dim=1)

            equals = predicted == labels.data

            print(equals.float().mean())


def save_model(checkpoint_name, model, optimizer, image_datasets, arch):
    # create a checkpoint         
    checkpoint = {'n_in': 25088,
                 'n_out': 102,
                 'n_h': [4096, 800],
                 'state_dict': model.state_dict(),
                  'arch': arch,
                 'epochs': 7,
                 'optimizer_state.dict': optimizer.state_dict(),
                 'class_to_idx:': image_datasets.class_to_idx}
    
    
    torch.save(checkpoint, checkpoint_name)
    
def calculate_acc(model, data):
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            #obtain outputs from the model
            outputs = model.forward(inputs)
            
            #get the probabilities
            _, predicted = outputs.max(dim=1)

            equals = predicted == labels.data

            print(equals.float().mean())

           

    
    
    
    
    
    
    
