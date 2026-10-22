
import numpy as np
from collections import Counter
from dataset import ADNISliceEntry, AD_LABEL, NC_LABEL, get_patient_labels


def _entry_slice_index(entry: ADNISliceEntry):
    """
    Extract the slice index of ADNI slice entry
    e.g. ~/.../.../396531_78.jpeg -> 78
    """
    return int(entry.img_path.stem.split("_")[1])


def get_split_stats(split: str, entries: list[ADNISliceEntry]):
    """
    Return statistics for entires corresponding to provided split

    Args:
        split   : whether we're looking at "train" or "test" data
        entries : the slice entries corresponding with the given split
    """
    patient_counts = Counter(e.patient_id for e in entries)
    patient_labels = get_patient_labels(entries)

    # Number of patients / recordings
    output = {
        "split": split,
        "entries": len(entries),
        "patients": len(patient_counts.keys()),
    }

    # AD vs NC patient counts
    for lbl, lbl_name in ((AD_LABEL, "AD"), (NC_LABEL, "NC")):
        patients = [p for p, l in patient_labels if l == lbl]
        output[f"{lbl_name}_patients"] = len(patients)
        output[f"{lbl_name}_slices"] = sum(patient_counts[p] for p in patients)

    # Distribution of MRI slices amongst patients
    counts = np.array(list(patient_counts.values()))
    output["slices_per_patient"] = {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": round(float(counts.mean()), 2),
        "std": round(float(counts.std()), 2),
    }

    # Range of 'slicde indexes' found throughout data
    idx = [_entry_slice_index(e) for e in entries]
    output["slice_index_range"] = [int(min(idx)), int(max(idx))]
    
    return output


if __name__ == "__main__":
    ...