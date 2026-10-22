
import os
from pathlib import Path
from dataclasses import dataclass
from typing import get_args, Literal, Callable, Union

import numpy
import numpy.typing
import sklearn.model_selection
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("/home/groups/comp3710/ADNI/AD_NC")
IMAGE_EXTENSION = ".jpeg"

NC_LABEL = 0 # Normal Control
AD_LABEL = 1 # Alzheimer's Disease


@dataclass(slots=True)
class ADNISliceEntry:
    patient_id: int
    label: int
    image_path: Path


class ADNIDataset(Dataset):
    """ADNI Dataset"""

    def __init__(self, entries: list[ADNISliceEntry]) -> None:
        ...

    def __len__(self):
        ...
  
    def __getitem__(self, idx: int):
        ...
        

def _patient_id_from_path(image_path: Path) -> int:
    """
    Extract patient ID from MRI image filename / path
    e.g. ~/.../.../396531_78.jpeg -> 396531

    image_path (Path) : the path to the given image
    """
    try:
        return int(image_path.stem.split("_", maxsplit=1)[0])
    except ValueError as err:
        raise ValueError(
            f"Couldn't extract patient ID from: {image_path.name}"
        ) from err
    

def _build_patient_class_mapping(root: Path, split: str) -> dict[int, int]:
    """
    Building a mapping between patient IDs and their diagnoses 

    root  (Path) : directory path to data
    split (str)  : are we accessing "train" or "test" data
    """

    patients: dict[int, int] = {}
  
    for cls_name, cls_lbl in (("AD", AD_LABEL), ("NC", NC_LABEL)):
        data_dir = root / split / cls_name 

        if not data_dir.is_dir():
            raise ValueError(f"Missing directory: {class_dir}")
    
        for path in data_dir.iterdir():
            if not path.is_file() or (path.suffix.lower() != IMAGE_EXTENSION):
                continue

            patient_id = patient_id_from_path(path)

            if patient_id in patients:
                if patients[patient_id] != cls_lbl:
                    raise ValueError("Patient found with both AD and NC classifications.")
                continue

            patients[patient_id] = cls_lbl

    return patients

def _generate_patient_ADNI_entries(root: Path, split: str, patient_id: in):
    entries: list[ADNISliceEntry] = []


def generate_datasets(root: Path, validation_split: float):

    if not 0 <= validation_split <= 1:
        raise ValueError("validation_split must be between 0 and 1.") 

    train_patients = _build_patient_class_mapping(root, "train")
    test_patients  = _build_patient_class_mapping(root, "test")

    overlap = train_patients.keys() & test_patients.keys()
    if overlap:
        raise ValueError("Patient overlap found between train and test data.")


    if not validation_split:
        ds_train = ADNIDataset(train_patients)
        ds_val   = None
        return ds_train, ds_val, ds_test

    patient_ids = np.array(list(train_patients))
    labels = np.array([train_patients[pid] for pid in patient_ids])

    # Splitting up list of unique patient IDs
    # Relevant ADNISliceEntries will be collated and put into datasets afterwards    
    train_ids, val_ids = sklearn.model_selection.train_test_split(
        patient_ids,
        test_size=validation_split,
        random_state=random_state,
        stratify=labels,
    )

    ds_train = ADNIDataset()
    ds_val   = ADNIDataset()

    return ds_train, ds_val, ds_test
  