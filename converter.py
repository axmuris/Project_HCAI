import os
import re
import shutil

rootdir = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/training/"
gt_rootdir = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/ground_truth/"
pred_rootdir = "/media/jonathan.bouyer/Transcend/5ETI/Projet/Resources/prediction/"

for root, subdirs, files in os.walk(rootdir):
    for file in files:
        if re.match(r"patient.*_frame.*.nii.gz", file):
            old_name = os.path.join(root, file)
            print(old_name)
            if "_gt" in file:
                new_name = os.path.join(gt_rootdir, file.replace("_gt", ""))
                shutil.copy(old_name, new_name)
            else:
                new_name = os.path.join(pred_rootdir, file)
                shutil.copy(old_name, new_name)