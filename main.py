from enum import Enum

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.data import Dataset
"""

from data import FeatureTypes, Frame, MaskValues



def preprocess_data(patient_list, *, kept_types=[], kept_part, kept_frame, kept_number, thr):
    """
    patient_list : list of PatientFeatures object whose features must be extracted to be fed to the SVM
    kept_types : list of FeatureTypes listing which type of features are kept
    kept_number : list of number of features kept in each of the listed types
    """
    assert len(kept_types) == len(kept_number), "The list of kept types must be of same length as the list of number of features kept in each of these types"

    features = []
    for patient in patient_list:
        for i, feature_type in enumerate(kept_types):
            features.append(patient.features[feature_type])



def svm():
    X = [
        [1, 0],
        [2, 1],
        [-1, 0],
        [0, 2],
        [0,-2]
    ]
    Y = [-1, -1, 1, 1, -1]

    penalty = 'l1'
    SVM = LinearSVC(penalty=penalty, loss='squared_hinge', dual=False)

    # Ajout d'une étape de pre-processing : normalisation
    pipeline = make_pipeline(StandardScaler(), SVM)

    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, Y, cv=loo)
    print(scores)


"""
class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.GaussianNoise(0.1)
        )

        nb_hidden_layers = 4

        for _ in range(nb_hidden_layers - 1):  # Adding 3 more hidden layers
            self.layers.add_module('hidden_linear', nn.Linear(32, 32))
            self.layers.add_module('hidden_batchnorm', nn.BatchNorm1d(32))
            self.layers.add_module('hidden_leakyrelu', nn.LeakyReLU())
            self.layers.add_module('hidden_gaussiannoise', nn.GaussianNoise(0.1))

        # Output layer
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.layers(x)
        output = self.output_layer(x)
        return output
    

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

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
    

def mlp():
    classes = ["NOR", "MINF", "DCM", "HCM", "ARV"]
    input_size = 
    output_size = len(classes)

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
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr0)

    nb_epoch = 400
    for epoch in range(nb_epoch):
        # potentially decrease lr 
        # scheduler.step()
        lr *= 0.97
        optimizer.lr = lr
        
        # Train : 1 epoch <-> loop once one the entire training dataset
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
"""

if __name__ == "__main__":
    svm()
