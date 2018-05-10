import argparse
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch 
import torchvision
from torch.autograd import Variable

# torch modlues
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

# HEALPix module for python
import healpy as hp

def str2bool(v):
    '''adopted from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse'''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Please proved boolean value for use_gpu.')

class HEALPixTransform(object):
    '''convert a square PIL img to a HEALPix array in numpy'''
    def __init__(self, nside):
        self.nside = nside 

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return self.img2healpix(img, self.nside)
        else:
            return self.img2healpix(np.array(img), self.nside)
        
    def cart_healpix(self, cartview, nside):
        '''read in an matrix and return a healpix pixelization map'''
        # Generate a flat Healpix map and angular to pixels
        healpix  = np.zeros(hp.nside2npix(nside), dtype=np.double)
        hptheta  = np.linspace(0, np.pi, num=cartview.shape[0])[:, None]
        hpphi    = np.linspace(-np.pi, np.pi, num=cartview.shape[1])
        pix = hp.ang2pix(nside, hptheta, hpphi)

        # re-pixelize
        healpix[pix] = np.fliplr(cartview)
        return healpix
    
    def ring2nest(self, healpix):
        nest = np.zeros(healpix.shape)
        ipix = hp.ring2nest(nside=hp.npix2nside(nest.shape[-1]), 
                            ipix=np.arange(nest.shape[-1]))
        nest[ipix] = healpix
        return nest
    
    def img2healpix(self, digit, nside):
        '''
        padding squre digits to 1*2 rectangles and
        convert them to healpix with a given nside
        '''
        h, w = digit.shape
        img = np.zeros((h, 2 * h))
        img[:, h - w // 2 : h - w // 2 + w] = digit
        return self.ring2nest(self.cart_healpix(img, nside))

class ToTensor(object):
    '''convert ndarrays to Tensors'''
    
    def __call__(self, healpix):
        return torch.from_numpy(healpix).double()

class Normalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self, tensor):
        tensor.div_(255)#.sub_(self.mean
#            ).div_(self.std).clamp_(min=0., max=1.
#            )
        return tensor.unsqueeze(0)

class HEALPixNet(nn.Module):
    
    def __init__(self):
        super(HEALPixNet, self).__init__()
        # nside = 8 -> nside = 4
        self.conv1 = nn.Conv1d(1,  32, 4, stride=4)
        
        # nside = 4 -> nside = 2
        self.conv2 = nn.Conv1d(32, 64, 4, stride=4)
        
        # nside**2 * 12 = 48
        self.fc1   = nn.Linear(48 * 64, 128)
        self.fc2   = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 48 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return x

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(history['train_loss'], label='train loss')
    ax[0].plot(history['val_loss'],   label='val   loss')
    ax[0].set_xlabel('epochs')
    ax[1].plot(history['train_acc'], label='train acc')
    ax[1].plot(history['val_acc'],   label='val   acc')
    ax[1].set_xlabel('epochs')
    ax[0].legend()
    ax[1].legend()
    plt.show()

def main(args):
    # dataloader
    batch_size = args.batch_size
    num_epochs = args.epochs
    use_gpu    = args.use_gpu

    transet = MNIST(root='./mnist', train=True, download=True)
    data_transforms = {
        'train' : transforms.Compose([
    #        transforms.RandomHorizontalFlip(),
            HEALPixTransform(nside=8),
            ToTensor(),
            Normalize(0.083, 0.254) # mnist setting
        ]),
        'val'   : transforms.Compose([
            HEALPixTransform(nside=8),
            ToTensor(),
            Normalize(0.083, 0.254) # mnist setting
        ]),
    }

    data_loaders = {
        'train' : torch.utils.data.DataLoader(
            MNIST(root='./mnist', train=True, download=True, 
                transform=data_transforms['train']),
            batch_size=batch_size, shuffle=True,
        ),
        'val'   : torch.utils.data.DataLoader(
            MNIST(root='./mnist', train=False,
                transform=data_transforms['val']),
            batch_size=batch_size, shuffle=True,
        ),
    }

    # get a instance of HEALPixNet
    net = HEALPixNet()
    net = net.double() # using double precision

    # optimizaer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    t = time.time()
    best_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    history = {'train_loss' : [], 
            'val_loss'   : [],
            'train_acc'  : [],
            'val_acc'    : []}


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('=' * 10)
        
        for phase in ['train', 'val']:
            running_loss     = 0.
            running_corrects = 0
            
            if phase == 'train':
                scheduler.step()
                net.train(True)
            else:
                net.train(False)
            
            for data in data_loaders[phase]:
                inputs, labels = data
                
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                
                optimizer.zero_grad()
                
                outputs  = net(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss     = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.data.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc  = running_corrects.float() / len(data_loaders[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(net.state_dict())
                
            # saving history
            history['{}_loss'.format(phase)].append(epoch_loss)
            history['{}_acc'.format(phase)].append(epoch_acc)        
                
        print()    

    time_elapsed = time.time() - t 
    print('Complete in {:.0f}min {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Top Accuracy: {:4f}'.format(best_acc))

    # Just make you feel you are doing things on spheres
    plt.figure(figsize=(8, 8))
    inputs, classes = next(iter(data_loaders['train']))
    pred = torch.max(net(inputs), 1)[1]
    for i in range(25):
        hp.orthview(inputs[i].numpy()[0, :], nest=True, sub=(5, 5, i + 1), half_sky=1, title=str(pred[i].item()), cbar=0)
    plt.show()

    net.load_state_dict(best_wts)
    plot_history(history)
    torch.save(net, 'net.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=str2bool, default=False,
                        help='GPU or not.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='# of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
                        
    args = parser.parse_args()                        
    main(args)