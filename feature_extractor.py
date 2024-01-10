import os

import SimpleITK as sitk

from radiomics import featureextractor

import numpy as np

import re

import pandas as pd

from matplotlib import pyplot as plt

from config_reader import extract_data_from_file
from data import PatientFeatures, FeatureTypes, feature_types_names, MaskValues, Frame, Feature



#Remove warnings an useless messages
sitk.ProcessObject_GlobalWarningDisplayOff()
import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)



#set extraction's parameters
def set_extractor_parameters():
    paramsFile = "./Params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

    return extractor

#Features extraction from an image and its mask
def feature_extractor (image, mask, extractor, patient, frame, mask_type):
    featureVector = extractor.execute(image, mask)
    
    for featureName in featureVector:
        if featureName.startswith("diagnostics"):
            continue

        featureValue = featureVector[featureName]
        featureValue = float(featureValue)  # Fix for 0-dimensional ndarrays

        feature_type_string, name = re.fullmatch(r"original_(.*)_(.*)", featureName).groups()
        feature = Feature(name, feature_types_names.inverse[feature_type_string], mask_type, frame, featureValue)
        patient.features.append(feature)
        #print("Computed %s: %s" % (featureName, featureVector[featureName]))

#extract features from all images from a file
def feature_extractor_folder (folder_path):
    extractor = set_extractor_parameters()
    listcase = os.listdir(folder_path)
    patients = []
    for case in listcase:
        if not case.startswith('patient'):
                continue
        print(case)
        patient = PatientFeatures()
        patient.nb = int(case[-3:])
        #Extract value from cfg
        
        valid, frameED, frameES, group, height, nb_frame, weight = extract_data_from_file(os.path.join(folder_path, case,'Info.cfg'))
        if not valid:
                print(f"Skipping {case}")
                continue
        
        height_feature = Feature("height", feature_types_names[FeatureTypes.GENERAL], None, None, height)
        weight_feature = Feature("weight", feature_types_names[FeatureTypes.GENERAL], None, None, weight)
        ed_es_feature = Feature("ED_ES_duration", feature_types_names[FeatureTypes.GENERAL], None, None, np.abs(frameED-frameES))
        patient.features += [height_feature, weight_feature, ed_es_feature]
        
        # Add leading 0 if necessary
        frames = {
                Frame.ED: format(frameED, '02d'),
                Frame.ES: format(frameES, '02d')
        }

        for frame in frames:
            image_path = os.path.join(folder_path, case, case +'_frame' + frames[frame] + '.nii.gz')
            mask_path = os.path.join(folder_path, case, case +'_frame' + frames[frame] + '_gt.nii.gz')

            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            image = sitk.Cast(image, sitk.sitkUInt8)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            mask.SetDirection(image.GetDirection())
            mask.SetOrigin(image.GetOrigin())            

            if image is None or mask is None:    # Something went wrong, in this case PyRadiomics will also log an error
                print('Error getting testcase!')
                exit()

            for mask_value in MaskValues:
                value = mask_value.value
                mask_array = sitk.GetArrayFromImage(mask)
                mask_array_part = np.zeros(mask_array.shape)
                mask_array_part[np.where(mask_array==value)] = 1
                mask_part= sitk.GetImageFromArray(mask_array_part)

                mask_part = sitk.Cast(mask_part, sitk.sitkUInt8)
                mask_part.SetDirection(image.GetDirection())
                mask_part.SetOrigin(image.GetOrigin())            
                mask_part.SetSpacing(image.GetSpacing())

                if mask_part is None:    # Something went wrong, in this case PyRadiomics will also log an error
                    print('Error getting testcase!')
                    exit()

                feature_extractor(image, mask_part, extractor, patient, frame, mask_value)
        patients.append(patient)
    return patients


def feature_extractor_folder_to_csv(folder_path, target_path):
    patients = feature_extractor_folder(folder_path)
    features_table = np.zeros((len(patients[0].features), len(patients)))
    feature_names = []
    for i, patient in enumerate(patients):
        for j, feature in enumerate(patient.features):
            features_table[j][i] = feature.value

            if i == 0:
                if feature.frame is None:
                    feature_name = feature.name
                else:
                    frame = "ED" if feature.frame == Frame.ED else "ES"
                    part = "RV" if feature.part == MaskValues.RV else ("LV" if feature.part == MaskValues.LV else "MYO")
                    feature_name = "_".join([feature.name,frame,part])
                feature_names.append(feature_name)
    df = pd.DataFrame(data=features_table, index=feature_names, columns=list(range(1,len(patients)+1)))
    df.to_csv(target_path, sep=";")


if __name__ == "__main__":
    folder_path = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/temp"
    feature_extractor_folder_to_csv(folder_path, "./features.csv")
