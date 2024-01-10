import os
import configparser

import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import numpy as np

import re

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
        
        valid, frameED, frameES, patient.group, patient.height, patient.nb_frame, patient.weight = extract_data_from_file(os.path.join(folder_path, case,'Info.cfg'))
        if not valid:
                print(f"Skipping {case}")
                continue
        patient.ED_ES_Duration=np.abs(frameED-frameES)
        
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


if __name__ == "__main__":
    folder_path = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/training"
    feature_extractor_folder(folder_path)
