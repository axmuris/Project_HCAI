import os
import re


def extract_data_from_file(file_path):
    """ Retourne un bool indiquant si l'extraction a fonctionné et si oui retourne les frames d'ED, ES, taille du patient, nb. de frames et poids du patient """
    valid = False
    ED, ES, group, height, nb_frame, weight = None, None, None, None, None, None
    with open(file_path) as file:
        content = file.read()
        
        match = re.fullmatch(r"ED: ([0-9]{1,2})\nES: ([0-9]{1,2})\nGroup: ((?:DCM|NOR))\nHeight: ([0-9]{3}(?:\.0)?)\nNbFrame: ([0-9]{2})\nWeight: ([0-9]{2,3}(?:\.0)?)\n", content)
        assert match, "Wrong format"
        
        ED, ES, group, height, nb_frame, weight = match.groups()
        ED = int(ED)
        ES = int(ES)
        height = float(height)
        nb_frame = int(nb_frame)
        weight = float(weight)

        valid = True
    return valid, ED, ES, group, height, nb_frame, weight
