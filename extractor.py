import os
import re
import shutil


def main():
    rootdir = "/media/jonathan.bouyer/Transcend/5ETI/Projet/"

    dic = {}

    for subdir, dirs, files in os.walk(rootdir):
        if "patient" in subdir:
            for file in files:
                if file == "Info.cfg":
                    patient_nb = subdir.split('/')[-1]
                    patient_nb = int(patient_nb[7:])
                    file_path = os.path.join(subdir, file)
                    lines = open(file_path).readlines()
                    height = lines[3]
                    weight = lines[5]
                    group = lines[2]

                    dic[patient_nb] = [int(float(height.split(': ')[1].strip())), int(float(weight.split(': ')[1].strip())), group.split(': ')[1].strip()]
                    
    for i in range(1, 151):
        print(f"{dic[i][0]}\t{dic[i][1]}\t{dic[i][2]}")
                        

if __name__ == "__main__":
    main()
