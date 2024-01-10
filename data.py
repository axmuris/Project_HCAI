from enum import Enum
from bidict import bidict


class FeatureTypes(Enum):
    GENERAL = 1
    SHAPE = 2
    INTENSITY = 3
    GLCM = 4
    GLRLM = 5
    GLSZM = 6
    ALL = 7


feature_types_names = bidict({
    FeatureTypes.SHAPE: "shape",
    FeatureTypes.INTENSITY: "firstorder",
    FeatureTypes.GLCM: "glcm",
    FeatureTypes.GLRLM: "glrlm",
    FeatureTypes.GLSZM: "glszm"
})


class MaskValues(Enum):
    RV = 0
    MYO = 1
    LV = 2


class Frame(Enum):
    ED = 0
    ES = 1


class Feature:
    def __init__(self, name_arg, type_arg, part_arg, frame_arg, value_arg):
        self.name = name_arg
        self.type = type_arg  # FeatureTypes
        self.part = part_arg  # MaskValues: LV, RV, MYO
        self.frame = frame_arg  # Frame: ED or ES
        self.value = value_arg
        self.importance = 0


class PatientFeatures:
    def __init__(self):
        self.nb = None
        self.features = []  # List of Feature objects
