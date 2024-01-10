import os
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def extract_frame_number(pred_path):
    start_index = pred_path.index('frame') + len("frame")
    return pred_path[start_index: start_index+2]

def visu(pred_path, gt_path):
    nii_img_pred  = nib.load(pred_path)
    nii_data_pred = nii_img_pred.get_fdata()

    nii_img_gt  = nib.load(gt_path)
    nii_data_gt = nii_img_gt.get_fdata()

    number_of_slices = nii_data_pred.shape[2]

    fig, ax = plt.subplots(2, number_of_slices, constrained_layout=True)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    for slice in range(number_of_slices):
        pred = nii_data_pred[:,:,slice]
        ax[0, slice].imshow(pred,cmap='gray', interpolation=None)
        ax[0, slice].set_title(f"layer {slice}")
        ax[0, slice].axis('off')

        gt = nii_data_gt[:,:,slice]
        ax[1, slice].imshow(gt,cmap='gray', interpolation=None)
        ax[1, slice].set_title(f"layer {slice}")
        ax[1, slice].axis('off')

    plt.show() 


def main():
    patient = 101
    assert 0 < patient < 151, "Invalid patient number"

    rootdir = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/"
    subdirs = ["training", "testing"]

    patientdir = "patient" + "0" * (3 - len(str(patient))) + str(patient)

    for subdir in subdirs:
        subroot = os.path.join(rootdir, subdir)
        if patientdir in os.listdir(subroot):
            path = os.path.join(rootdir, subdir, patientdir)
            pred_files = [file for file in os.listdir(path) if re.match(patientdir + r"_frame...nii.gz", file)]
            gt_files = [file[:file.index(".nii.gz")] + "_gt.nii.gz" for file in pred_files]
            
            visu(os.path.join(path, pred_files[0]), os.path.join(path, gt_files[0]))
            visu(os.path.join(path, pred_files[1]), os.path.join(path, gt_files[1]))
        
            
if __name__ == "__main__":
    main()
