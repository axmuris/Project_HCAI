import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


classes = ["NOR", "MINF", "DCM", "HCM", "RV"]
class_to_int = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}
int_to_class = {0: "NOR", 1: "MINF", 2: "DCM", 3: "HCM", 4: "RV"}



class CustomFeaturesDataset(Dataset):
    def __init__(self, features_csv_file_path, groups_csv_file_path):
        self.features = pd.read_csv(features_csv_file_path, header=0, index_col=0, sep=";").T.values
        self.groups = pd.read_csv(groups_csv_file_path, header=None, index_col=0, sep=";").values.ravel()
        self.groups = np.array([class_to_int[group] for group in self.groups])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.features[idx], self.groups[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.name = "mlp"

        self.input_layer = nn.Linear(input_size, 32)
        self.dropout = nn.Dropout(0.25)
        self.nb_hidden_layers = 4
        self.linear = nn.Linear(32, 32)
        self.batch_norm = nn.BatchNorm1d(32)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.to(torch.float32)
        x = nn.functional.normalize(x)

        #if self.training:
        #    x = self.dropout(x)

        x = self.input_layer(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        if self.training:
            x += (0.05**0.5)*torch.randn(x.shape)

        for _ in range(self.nb_hidden_layers - 1):
            x = self.linear(x)
            x = self.batch_norm(x)
            if self.training:
                x += (0.05**0.5)*torch.randn(x.shape)
            x = self.leaky_relu(x)

        output = self.output_layer(x)
        output = self.softmax(output)

        return output
    
class MLPClassifier2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.name = "mlp"


        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        
        self.nb_hidden_layers = 4

        for i in range(self.nb_hidden_layers - 1):  # Adding 3 more hidden layers
            self.layers.add_module('hidden_linear', nn.Linear(32, 32))
            self.layers.add_module('hidden_batchnorm', nn.BatchNorm1d(32))
            self.layers.add_module('hidden_leakyrelu', nn.LeakyReLU())
            
        self.softmax = nn.Softmax()
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.to(torch.float32)
        x = nn.functional.normalize(x)
        x = self.layers(x)

        output = self.output_layer(x)
        output = self.softmax(output)

        return output
    
class MLPClassifier2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.name = "mlp"


        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        
        self.nb_hidden_layers = 4

        for i in range(self.nb_hidden_layers - 1):  # Adding 3 more hidden layers
            self.layers.add_module('hidden_linear', nn.Linear(32, 32))
            self.layers.add_module('hidden_batchnorm', nn.BatchNorm1d(32))
            self.layers.add_module('hidden_leakyrelu', nn.LeakyReLU())
            
        self.softmax = nn.Softmax()
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.to(torch.float32)
        x = nn.functional.normalize(x)
        x = self.layers(x)

        output = self.output_layer(x)
        output = self.softmax(output)

        return output


def get_subset_indices(max_idx, nb_features):
    return random.sample(list(range(max_idx)), nb_features)


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
    output_size = 5
    
    generator = torch.Generator().manual_seed(42)
    training_data = CustomFeaturesDataset("./data/shape_features_training.csv", "./data/groups_training.csv")
    test_data = CustomFeaturesDataset("./data/shape_features_testing.csv", "./data/groups_testing.csv")
    """
    val_data = CustomFeaturesDataset("./data/features_50_sub_val.csv", "./data/groups_sub_val.csv")

    """

    test_data, val_data = random_split(test_data, [0.5, 0.5], generator=generator)
    train_dataloader = DataLoader(training_data, batch_size=25, shuffle=True)
    validation_dataloader = DataLoader(val_data, batch_size=25, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=25, shuffle=True)
    

    if torch.cuda.is_available():
        print('use cuda')
        device = torch.device('cuda')
    else:
        print('use CPU')
        device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss()

    nb_mlp = 20
    nb_features = int(0.5 * input_size)
    plot_lines = 5
    plot_col = 10
    fig, axs = plt.subplots(plot_lines, plot_col)

    nb_epoch = 400
    nb_rep = 2
    epoch_patience = 40
    epoch_stability = 0
    early_stop_thr = 2.5e-2
    epoch_thr = 200

    MLPs = []
    subsets = []

    for i in range(nb_mlp):
        print(f"\n\n\nMLP {i}\n\n\n")

        net = MLPClassifier2(nb_features, output_size)
        init_params(net)
        net = net.to(device)

        lr0 = 5e-4
        optimizer = torch.optim.Adam(net.parameters(), lr=lr0)

        subset_indices = get_subset_indices(input_size, nb_features)
        subsets.append(subset_indices)

        lossesTR    = []
        lossesTRAll = []
        losses = []
        lossesTE    = []
        idxEpoch    = [0]
        bestAcc     = 0

        lr = lr0

        for epoch in range(nb_epoch):
            # potentially decrease lr 
            #print(epoch, "/", nb_epoch)
            lr *= 0.97
            optimizer.lr = lr
            losses = []

            #print(lr)

            # Train : 1 epoch <-> loop once one the entire training dataset
            start = time.time()
            # Train : 1 epoch <-> loop once one the entire training dataset
            for _ in range(nb_rep):
                for inputs, targets in train_dataloader:
                    inputs = inputs[:, subset_indices]

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
            #print('Epoch : %d Train Loss : %.3f         time: %.3f' % (epoch, np.mean(losses),end-start))
            
            # Evaluate the current network on the validation dataset
            net.train()
            total = 0
            correct = 0
            losses = []
            start = time.time()

            for inputs, targets in validation_dataloader:
                inputs = inputs[:, subset_indices]
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.data.item())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
            end = time.time()
            lossesTE.append(np.mean(losses))
            if len(lossesTE) > 1 and epoch > epoch_thr and np.abs(lossesTE[-1] - lossesTE[-2])/lossesTE[-2] < early_stop_thr:
                epoch_stability += 1
                if epoch_stability >= epoch_patience:
                    print("EARLY STOPPING")
                    break
            else:
                epoch_stability = 0

            bestAcc = max(bestAcc,100.*correct.item()/total)
            #print('Epoch : %d Test Loss  : %.3f        Test Acc %.3f       time: %.3f' % (epoch, np.mean(losses),100.*correct/total,end-start))
            #print('--------------------------------------------------------------')
            net.train()
        """
        print ('----------------------------------------------------------------------------')
        print ('----------------------------------------------------------------------------')
        print ('----------------------------------------------------------------------------')
        print ('best accuracy      : '+str(bestAcc))
        print ('best loss on train : '+str(np.min(lossesTR)) + ' idx '+str(np.argmin(lossesTR)))
        print ('best loss on test  : '+str(np.min(lossesTE)) + ' idx '+str(np.argmin(lossesTE)))
        """
        MLPs.append(net)
        
        t = np.arange(0, len(lossesTRAll))

        axs[i // plot_col, i % plot_col].plot(t, lossesTRAll, 'b', idxEpoch[1:], lossesTR, 'ro', idxEpoch[1:], lossesTE, 'gs')

    test_loss = 0.0
    correct, total = 0,0
    y_true = []
    y_pred = []
    for data,label in test_dataloader:
        if device == torch.device('cuda'):
            data, label = data.cuda(), label.cuda()

        res = torch.from_numpy(np.zeros((test_dataloader.batch_size, 5)))
        for i, mlp in enumerate(MLPs):
            subset_indices = subsets[i]
            output = mlp(data[:, subset_indices])
            res += output

        for o,l in zip(torch.argmax(res,axis = 1),label):
            y_true.append(o)
            y_pred.append(l)
            if o == l:
                correct += 1
            total += 1
        loss = criterion(output,label)
        test_loss += loss.item() * data.size(0)
    print(correct, total, test_loss)
    print(classification_report(y_true, y_pred, target_names=classes))
    y_true = [int_to_class[y.data.item()] for y in y_true]
    y_pred = [int_to_class[y.data.item()] for y in y_pred]
    print(confusion_matrix(y_true, y_pred, labels=classes))
    plt.show()
        
if __name__ == "__main__":    
    main()
