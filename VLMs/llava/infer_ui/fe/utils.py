import pydicom
import numpy as np
import os
from tqdm import tqdm
import glob
import base64
import pickle

def split_volume(slices, volume):
    patient_height = float(slices[0].PatientSize) * 100 if hasattr(slices[0], 'PatientSize') else 170  # Default height = 170 cm
    slice_thickness = patient_height / volume.shape[0]
    
    head_neck_height = 0.25 * patient_height
    chest_height = 0.3 * patient_height

    head_neck_slices = int(head_neck_height / slice_thickness)
    chest_slices = int(chest_height / slice_thickness)
    
    overlap = 30 # Số lát cắt overlap

    head_neck = volume[:head_neck_slices + overlap, :, :]
    chest = volume[head_neck_slices - overlap:head_neck_slices + chest_slices + overlap, :, :]
    abdomen_pelvis = volume[head_neck_slices + chest_slices - overlap:, :, :]

    return head_neck, chest, abdomen_pelvis

def slices_to_volume(slices):
    return np.stack([s.pixel_array for s in slices], axis=0)

def load_and_sort_dicom(dicom_files, modality_prefix):
    print(dicom_files)
    slices = []

    for file in tqdm(dicom_files, desc=f"Loading {modality_prefix} DICOM files"):
        ds = pydicom.dcmread(file)
        slices.append(ds)

    # Sắp xếp theo vị trí slice location (nếu có) hoặc InstanceNumber
    slices.sort(key=lambda x: float(x.SliceLocation) if hasattr(x, 'SliceLocation') else x.InstanceNumber)
    slices.reverse()

    return slices

def encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(pickle.dumps(arr)).decode("utf-8")