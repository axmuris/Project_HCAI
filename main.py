from enum import Enum
import numpy as np

import pandas as pd

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.data import Dataset
"""

from feature_extractor import feature_extractor_folder
from data import FeatureTypes, Frame, MaskValues



def extract_feature_by_name(features_csv_path, feature_names, gt_csv_path):
    features_df = pd.read_csv(features_csv_path, header=0, index_col=0, sep=";")

    svm_entries = []
    for column_name in features_df:
        features = []
        for feature_name in feature_names:
            features.append(features_df.loc[feature_name][column_name])
        svm_entries.append(features)

    gt_df = pd.read_csv(gt_csv_path, index_col=0, sep=";")
    svm_data = [(svm_entries[i], gt_df.iloc[i][0]) for i in range(len(svm_entries))]

    return svm_data



def preprocess_data(patient_list, *, kept_types=[], kept_parts=[MaskValues.LV, MaskValues.RV, MaskValues.MYO],
                    kept_frames=[Frame.ED, Frame.ES], kept_number=None, thr=0):
    """
    patient_list : list of PatientFeatures object whose features must be extracted to be fed to the SVM
    kept_types : list of FeatureTypes listing which type of features are kept
    kept_parts: list of MaskValues kept for the SVM
    kept_frames: list of Frame kept for the SVM
    kept_number : max number of features kept
    thr: threshold of importance of the features
    """

    keep_all_types = kept_types == []
    features = []
    for patient in patient_list:
        nb = 0
        for feature in patient.features:
            valid_type = feature.type in kept_types or keep_all_types
            valid_part = feature.part in kept_parts
            valid_frame = feature.frame in kept_frames
            valid_importance = feature.importance >= thr
            if valid_type and valid_part and valid_frame and valid_importance:
                features.append(feature.value)
                nb += 1
                if kept_number is not None and nb > kept_number:
                    break
    return features



def svm(X, Y):
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    penalty = 'l1'
    SVM = LinearSVC(penalty=penalty, loss='squared_hinge', dual=False)

    # Ajout d'une étape de pre-processing : normalisation
    pipeline = make_pipeline(StandardScaler(), SVM)

    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, Y, cv=loo)
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


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

def main():
    folder_path = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/temp"
    patients_list = feature_extractor_folder(folder_path)
    features = preprocess_data(patients_list, kept_types=[FeatureTypes.SHAPE])
    print(features)


if __name__ == "__main__":
    features = extract_feature_by_name("./data/features_training.csv",
                                  ["VoxelVolume_ED_MYO", "SurfaceVolumeRatio_ES_LV", "LeastAxisLength_ES_LV",
                                   "Maximum2DDiameterColumn_ED_LV","Maximum2DDiameterRow_ED_LV","Maximum2DDiameterSlice_ED_LV",
                                   "Maximum3DDiameter_ES_RV", "Maximum3DDiameter_ES_MYO", "SurfaceArea_ED_RV", "height"],
                                   "./data/groups.csv")
    X = [x for x, _ in features]
    Y = [y for _, y in features]

    svm(X, Y)
    #print(extract_feature_by_name("./data/features_training.csv",
    #                              ["VoxelVolume_ED_MYO", "SurfaceVolumeRatio_ES_LV", "LeastAxisLength_ES_LV",
    #                               "Maximum2DDiameterColumn_ED_LV","Maximum2DDiameterRow_ED_LV","Maximum2DDiameterSlice_ED_LV",
    #                               "Maximum3DDiameter_ES_RV", "Maximum3DDiameter_ES_MYO", "SurfaceArea_ED_RV", "height"],
    #                               "./data/groups.csv"))
