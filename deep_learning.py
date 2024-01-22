from enum import Enum
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import random

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix
#import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# if torch.cuda.is_available():
#     print('use cuda')
#     device = torch.device('cuda')
# else:
#     print('use CPU')
#     device = torch.device('cpu')
device = torch.device('cpu')
    
classes = ["NOR", "MINF", "DCM", "HCM", "RV"]
class_to_int = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}
int_to_class = {0: "NOR", 1: "MINF", 2: "DCM", 3: "HCM", 4: "RV"}

# class_to_int = {"NOR": 0, "MINF": 0, "DCM": 1, "HCM": 0, "RV": 0}
# int_to_class = {0: "NOR", 0: "MINF", 1: "DCM", 0: "HCM", 0: "RV"}


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.name = "mlp"

        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.dropout=nn.Dropout(0.2)

        nb_hidden_layers = 4

        for _ in range(nb_hidden_layers - 1):  # Adding 3 more hidden layers
            self.layers.add_module('hidden_linear', nn.Linear(32, 32))
            self.layers.add_module('hidden_batchnorm', nn.BatchNorm1d(32))
            self.layers.add_module('hidden_leakyrelu', nn.LeakyReLU())

        # Output layer
        #self.output_layer = nn.Sigmoid()
        self.output_layer = nn.Linear(32,output_size)
        #self.output_layer.add_module('linear',nn.Linear(32,output_size))

    def forward(self, x):
        # if self.training:
        #     x=self.dropout(x)
        x = self.layers(x.to(torch.float32))
        x += (0.1**0.5)*torch.randn(x.shape)
        output = self.output_layer(x)
        return output
    

class CustomFeaturesDataset(Dataset):
    def __init__(self, features_csv_file_path, groups_csv_file_path):
        self.features = pd.read_csv(features_csv_file_path, header=0, index_col=0, sep=";").T.values
        #self.features = self.extractrdmfeatures()
        self.groups = pd.read_csv(groups_csv_file_path, header=None, index_col=0, sep=";").values.ravel()
        self.groups = [class_to_int[group] for group in self.groups]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.features[idx], self.groups[idx]

    def extractrdmfeatures(self):

        #feat_temp=np.empty()

        return 


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
    input_size = 87
    output_size = len(classes)
    
    training_data = CustomFeaturesDataset(r"C:\Users\Axel\Desktop\Ecole\5A_CPE\projet\Project_HCAI\data\new_features_training.csv", r"C:\Users\Axel\Desktop\Ecole\5A_CPE\projet\Project_HCAI\data\groups_training.csv")
    test_data = CustomFeaturesDataset(r"C:\Users\Axel\Desktop\Ecole\5A_CPE\projet\Project_HCAI\data\new_features_testing.csv", r"C:\Users\Axel\Desktop\Ecole\5A_CPE\projet\Project_HCAI\data\groups_testing.csv")
    #validation_data = CustomFeaturesDataset(r"C:\Users\Axel\Desktop\Ecole\5A_CPE\projet\Project_HCAI\data\features_50_sub_val.csv", r"C:\Users\Axel\Desktop\Ecole\5A_CPE\projet\Project_HCAI\data\groups_sub_val.csv")
  
    generator1 = torch.Generator().manual_seed(42)
    test_data,validation_data = random_split(test_data, [0.5,0.5], generator=generator1)

    train_dataloader = DataLoader(training_data, batch_size=25, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=25, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=25, shuffle=True)
    

    net = MLPClassifier(input_size, output_size)
    init_params(net)
    net = net.to(device)

    dlossesTR    = {}
    dlossesTRAll = {}
    dlossesTE    = {}
    didxEpoch    = {}
    dbestAcc     = {}
    dnbParam     = {}

    lr0 = 5e-4

    test_name = net.name + '_lr0_' + "{:.1e}".format(lr0)
    print(test_name)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr0)
    criterion = nn.CrossEntropyLoss()

    nb_epoch = 500
    epoch_patience = 400
    epoch_stability = 0
    early_stop_thr = 1e-3

    lossesTR    = []
    lossesTRAll = []
    losses = []
    lossesTE    = []
    idxEpoch    = [0]
    bestAcc     = 0

    lr = lr0
    lr_gamma = 0.01
    lr_step = 50

    for epoch in range(nb_epoch):
        # potentially decrease lr 
        # scheduler.step()
        print(epoch, "/", nb_epoch)
        #lr *= 0.97
        lr = lr0 * lr_gamma**int(epoch/lr_step)
        optimizer.lr = lr

        print(lr)

        # Train : 1 epoch <-> loop once one the entire training dataset
        start = time.time()
        # Train : 1 epoch <-> loop once one the entire training dataset
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            #if batch_idx ==0:
                #continue
            #if cuda_available:
            #    inputs, targets = inputs.cuda(), targets.cuda()
            # transfer to GPU if avalaible


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
        net.eval()
        total = 0
        correct = 0
        losses = []
        start = time.time()

        for batch_idx, (inputs, targets) in enumerate(validation_dataloader):
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

        bestAcc = max(bestAcc,100.*correct/total)
        print('Epoch : %d Test Loss  : %.3f        Test Acc %.3f       time: %.3f' % (epoch, np.mean(losses),100.*correct/total,end-start))
        print('--------------------------------------------------------------')
        net.train()        



    test_loss = 0.0
    correct, total = 0,0

    data_stack=[]
    target_stack=[]

    for data,label in test_dataloader:
        if device == torch.device('cuda'):
            data, label = data.cuda(), label.cuda()
        output = net(data)

        for o,l in zip(torch.argmax(output,axis = 1),label):
            data_stack.append(o)
            target_stack.append(l)
            if o == l:
                correct += 1
            total += 1
        loss = criterion(output,label)
        test_loss += loss.item() * data.size(0)

    #data_stack=torch.Tensor(data_stack)
    #target_stack=torch.Tensor(target_stack)
    data_stack = [int_to_class[y.data.item()] for y in data_stack]
    target_stack = [int_to_class[y.data.item()] for y in target_stack]
    print(confusion_matrix(data_stack,target_stack,labels=classes))

    # to store losses for all epochs / iterations
    lossesTR
    dlossesTR[test_name]    = lossesTR
    dlossesTRAll[test_name] = lossesTRAll
    dlossesTE[test_name]    = lossesTE
    didxEpoch[test_name]    = idxEpoch
    dbestAcc[test_name]     = bestAcc
    dnbParam[test_name]     = count_parameters(net)

    for n in [test_name]: #[ 'mlp_lr0_5.0e-04' ]: # ajouter dans cette liste le nom des reseaux que vous tester 
        if n in dlossesTR:
            print ('----------------------------------------------------------------------------')
            print ('----------------------------------------------------------------------------')
            print ('----------------------------------------------------------------------------')
            print (n)
            print ('best accuracy      : '+str(dbestAcc[n].item()))
            print ('best loss on train : '+str(np.min(dlossesTR[n])) + ' idx '+str(np.argmin(dlossesTR[n])))
            print ('best loss on test  : '+str(np.min(dlossesTE[n])) + ' idx '+str(np.argmin(dlossesTE[n])))
            print ('n param            : '+str(dnbParam[n]))

            print(f'Testing Loss:{test_loss/len(test_dataloader)}')
            print(f'Testing accuracy: {correct / total}')

            # evenly sampled time at 200ms intervals
            t = np.arange(0, len(dlossesTRAll[n]))
            

            plt.plot(t, dlossesTRAll[n], 'b', didxEpoch[n][1:], dlossesTR[n], 'ro', didxEpoch[n][1:], dlossesTE[n], 'gs')
            plt.title(n)
            plt.show()

 
if __name__ == "__main__":
    main()