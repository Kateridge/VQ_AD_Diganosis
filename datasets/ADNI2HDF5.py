import h5py
import pandas as pd
import os
import nibabel as nib
import numpy as np
from monai.transforms import ScaleIntensity

dataroot = 'C:\\Users\\Lalal\\Projects\\datasets\\ADNIALL'
label_csv = pd.read_csv(os.path.join(dataroot, 'ADNI.csv'))
sub_list = label_csv['Sub'].to_list()
transform = ScaleIntensity()


def checkStr(data):
    if type(data).__name__ == 'str':
        if data == '>1700':
            return 1700
        if data == '<8':
            return 8
        if data == '<80':
            return 80
        if data == '<200':
            return 200
        if data == '>120':
            return 120
        if data == '>1300':
            return 1300
        return float(data)
    else:
        return data


def getTabularFromSeries(sub_info):
    # mapping str to int
    gender_dict = {'Female': 0, 'Male': 1}

    # Demographic information
    age = sub_info.Age
    gender = gender_dict[sub_info.Sex]
    edu = sub_info.Edu
    # Genetic information
    APOE4 = sub_info.APOE4
    # Cognitive tests
    MMSE = sub_info.MMSE
    ADAS11 = sub_info.ADAS11
    ADAS13 = sub_info.ADAS13
    RAVLT1 = sub_info.RAVLT1
    RAVLT2 = sub_info.RAVLT2
    RAVLT3 = sub_info.RAVLT3
    # Biomarkers at baseline
    abeta = sub_info.abeta
    abeta = checkStr(abeta)
    tau = sub_info.tau
    tau = checkStr(tau)
    ptau = sub_info.ptau
    ptau = checkStr(ptau)
    fdg = sub_info.fdg
    fdg = checkStr(fdg)

    tabular_list = [age, gender, edu, APOE4, MMSE, ADAS11, ADAS13, RAVLT1, RAVLT2, RAVLT3, abeta, tau, ptau, fdg]
    return tabular_list


# save data to hdf5
# -sub_id (RID, DX, missing)
# --tabular
# --MRI
# --FDG
# --Amyloid
# --Tau
with h5py.File('D:\\datasets\\ADNI_ALL\\ADNI.hdf5', 'w') as f:
    for _, sub_info in label_csv.iterrows():
        sub_id = sub_info.iloc[1]
        sub_group = f.create_group(sub_id)
        # get tabular information
        sub_tabular = getTabularFromSeries(sub_info)
        sub_hdf5_tabular = sub_group.create_dataset('tabular', dtype=np.float32, data=np.array(sub_tabular))
        # get label
        DX = sub_info.iloc[7]
        sub_group.attrs['DX'] = DX
        sub_group.attrs['RID'] = sub_id

        sub_missing = sub_info.iloc[2:7].to_list()
        print(f'processing {sub_id}')
        # get images
        if sub_missing[0]:
            MRI_np = nib.load(os.path.join(dataroot, sub_id, 'T1_brain.nii.gz')).get_fdata()
            MRI_np = transform(MRI_np[4:116, 5:141, 0:114])
            sub_hdf5_MRI = sub_group.create_dataset('MRI', dtype=np.float32, data=MRI_np)
        if sub_missing[1]:
            FDG_np = nib.load(os.path.join(dataroot, sub_id, 'FDG.nii.gz')).get_fdata()
            FDG_np = transform(FDG_np[4:116, 5:141, 0:114])
            sub_hdf5_FDG = sub_group.create_dataset('FDG', dtype=np.float32, data=FDG_np)
        if sub_missing[2]:
            Amyloid_np = nib.load(os.path.join(dataroot, sub_id, 'AV45.nii.gz')).get_fdata()
            Amyloid_np = transform(Amyloid_np[4:116, 5:141, 0:114])
            sub_hdf5_Amyloid = sub_group.create_dataset('Amyloid', dtype=np.float32, data=Amyloid_np)
        elif sub_missing[3]:
            Amyloid_np = nib.load(os.path.join(dataroot, sub_id, 'FBB.nii.gz')).get_fdata()
            Amyloid_np = transform(Amyloid_np[4:116, 5:141, 0:114])
            sub_hdf5_Amyloid = sub_group.create_dataset('Amyloid', dtype=np.float32, data=Amyloid_np)
        if sub_missing[4]:
            Tau_np = nib.load(os.path.join(dataroot, sub_id, 'Tau.nii.gz')).get_fdata()
            Tau_np = transform(Tau_np[44:116, 5:141, 0:114])
            sub_hdf5_Tau = sub_group.create_dataset('Tau', dtype=np.float32, data=Tau_np)

        sub_missing = sub_missing[0:2] + [sub_missing[2] + sub_missing[3]] + [sub_missing[4]]
        sub_group.attrs['missing'] = sub_missing

print('Done')
