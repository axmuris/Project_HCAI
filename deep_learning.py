import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
#import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


classes = ["NOR", "MINF", "DCM", "HCM", "RV"]
class_to_int = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}
int_to_class = {0: "NOR", 1: "MINF", 2: "DCM", 3: "HCM", 4: "RV"}


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.name = "mlp"

        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(0.2)

        nb_hidden_layers = 4

        for _ in range(nb_hidden_layers - 1):  # Adding 3 more hidden layers
            self.layers.add_module('hidden_linear', nn.Linear(32, 32))
            self.layers.add_module('hidden_batchnorm', nn.BatchNorm1d(32))
            self.layers.add_module('hidden_leakyrelu', nn.LeakyReLU())

        # Output layer
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layers(x.to(torch.float32))
        x += (0.1**0.5)*torch.randn(x.shape)
        output = self.output_layer(x)
        return output
    

class CustomFeaturesDataset(Dataset):
    def __init__(self, features_csv_file_path, groups_csv_file_path):
        self.features = pd.read_csv(features_csv_file_path, header=0, index_col=0, sep=";").T.values
        self.groups = pd.read_csv(groups_csv_file_path, header=None, index_col=0, sep=";").values.ravel()
        self.groups = [class_to_int[group] for group in self.groups]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.features[idx], self.groups[idx]


def count_parameters(model):
    #for parameter in model.parameters():
    #    print(parameter)
    #print ('nb of trainable parameters')
    return (sum([p.numel() for p in model.parameters() if p.requires_grad]))

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        #print(m)
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            #if m.bias:
            #init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            #if m.bias:
            init.constant_(m.bias, 0) 
    

def main():
    input_size = 531
    output_size = len(classes)

    training_data = CustomFeaturesDataset("./data/features_training.csv", "./data/groups_training.csv")
    test_data = CustomFeaturesDataset("./data/features_testing.csv", "./data/groups_testing.csv")

    train_dataloader = DataLoader(training_data, batch_size=20, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True)

    if torch.cuda.is_available():
        print('use cuda')
        device = torch.device('cuda')
    else:
        print('use CPU')
        device = torch.device('cpu')

    net = MLPClassifier(input_size, output_size)
    init_params(net)
    net = net.to(device)

    lr0 = 5e-4

    test_name = net.name + '_lr0_' + "{:.1e}".format(lr0)
    print(test_name)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr0)
    criterion = nn.CrossEntropyLoss()

    nb_epoch = 400
    epoch_patience = 40
    epoch_stability = 0
    early_stop_thr = 1e-3

    lossesTR    = []
    lossesTRAll = []
    losses = []
    lossesTE    = []
    idxEpoch    = [0]
    bestAcc     = 0

    lr = lr0

    for epoch in range(nb_epoch):
        # potentially decrease lr 
        print(epoch, "/", nb_epoch)
        lr *= 0.97
        optimizer.lr = lr
        losses = []

        print(lr)

        # Train : 1 epoch <-> loop once one the entire training dataset
        start = time.time()
        # Train : 1 epoch <-> loop once one the entire training dataset
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # clear gradient    
            optimizer.zero_grad()
            
            # convert input to Variable
            #inputs, targets = Variable(inputs), Variable(targets)
            
            # compute the output of the network for the given inputs
            outputs = net(inputs)
            
            # compute the loss function
            loss = criterion(outputs, targets)
            
            # compute the gradient w.r. to all weights 
            loss.backward()
            
            # one update of the parameter update
            optimizer.step()

            # store loss of the current iterate
            losses.append(loss.data.item())
            lossesTRAll.append(loss.data.item())

        end = time.time()
        # meanlosses = torch.mean(torch.stack(losses)) 
        lossesTR.append(np.mean(losses))
        idxEpoch.append(idxEpoch[-1] + len(losses))
        print('Epoch : %d Train Loss : %.3f         time: %.3f' % (epoch, np.mean(losses),end-start))
        
        # Evaluate the current network on the validation dataset
        net.train()
        total = 0
        correct = 0
        losses = []
        start = time.time()

        for inputs, targets in test_dataloader:
            # transfer to GPU if avalaible
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.data.item())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        end = time.time()
        lossesTE.append(np.mean(losses))
        if len(lossesTE) > 1 and np.abs(lossesTE[-1] - lossesTE[-2]) < early_stop_thr:
            epoch_stability += 1
            if epoch_stability >= epoch_patience:
                print("EARLY STOPPING")
                break

        bestAcc = max(bestAcc,100.*correct.item()/total)
        print('Epoch : %d Test Loss  : %.3f        Test Acc %.3f       time: %.3f' % (epoch, np.mean(losses),100.*correct/total,end-start))
        print('--------------------------------------------------------------')
        net.train()

    print ('----------------------------------------------------------------------------')
    print ('----------------------------------------------------------------------------')
    print ('----------------------------------------------------------------------------')
    print ('best accuracy      : '+str(bestAcc))
    print ('best loss on train : '+str(np.min(lossesTR)) + ' idx '+str(np.argmin(lossesTR)))
    print ('best loss on test  : '+str(np.min(lossesTE)) + ' idx '+str(np.argmin(lossesTE)))
    print ('n param            : '+str(count_parameters(net)))

    # evenly sampled time at 200ms intervals
    t = np.arange(0, len(lossesTRAll))

    plt.plot(t, lossesTRAll, 'b', idxEpoch[1:], lossesTR, 'ro', idxEpoch[1:], lossesTE, 'gs')
    plt.show()

if __name__ == "__main__":
    main()