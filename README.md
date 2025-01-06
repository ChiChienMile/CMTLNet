# CMTLNet
This repository provides a PyTorch implementation of the paper "Cooperative Multi-task Learning and Interpretable Image Biomarkers for Glioma Grading and Molecular Subtyping." (MedIA 2024)

## Data Acquisition

The following public datasets are utilized in this project:

- **REMBRANDT**: [http://doi.org/10.7937/K9/TCIA.2015.588OZUZB](http://doi.org/10.7937/K9/TCIA.2015.588OZUZB)
- **LGG-1p/19q**: [http://doi.org/10.7937/K9/TCIA.2017.DWEHTZ9V](http://doi.org/10.7937/K9/TCIA.2017.DWEHTZ9V)
- **UCSF-PDGM**: [http://doi.org/10.7937/tcia.bdgf-8v37](http://doi.org/10.7937/tcia.bdgf-8v37)

**Segmentation Data for LGG-1p/19q**:
- [http://dx.doi.org/10.17632/rssf5nxxby.1](http://dx.doi.org/10.17632/rssf5nxxby.1)

Additional datasets:
- **BraTS (MI-20)**: [http://braintumorsegmentation.org/](http://braintumorsegmentation.org/)
- **EGD Dataset**: Contains data from 774 patients, available at [https://xnat.bmia.nl/REST/projects/egd](https://xnat.bmia.nl/REST/projects/egd). Detailed descriptions are provided in the associated data publication.

## Data Preprocessing

### Step 1: Convert DICOM to NIfTI

Convert all DICOM files to NIfTI (`.nii`) format using the free software **MRIConvert**.

- **MRIConvert Download**: [https://idoimaging.com/programs/214](https://idoimaging.com/programs/214)

### Step 2: Download Registration Template

Download the **SRI24 atlas** registration template from the following DOI link:

- **SRI-atlas Download**: [https://doi.org/10.7937/9j41-7d44](https://doi.org/10.7937/9j41-7d44)

After downloading, ensure the file is named `spgr_unstrip_lps.nii`.

### Step 3: Skull Stripping and Image Registration

1. **Skull Stripping**:
   - Utilize **FSL's BET** tool to perform skull stripping on both the SRI-atlas and the NIfTI data obtained from Step 1.
   - **FSL Documentation**: [FSL Official Documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/)

2. **Image Registration**:
   - Use **ANTsPy** to register all skull-stripped T1 MRI data to the skull-stripped SRI-atlas.
   - Subsequently, register all skull-stripped T2-weighted MRI data to the registered T1 MRI.
   - **ANTsPy GitHub**: [https://github.com/ANTsX/ANTsPy](https://github.com/ANTsX/ANTsPy)

**Relevant Scripts and Data**:
- Registration code: `preprocessing/1_registration.py`
- Example data and skull-stripped SRI-atlas (`spgr_unstrip_lps_b.nii.gz`): `preprocessing/regdata/`

### Step 4: Crop and Pad Background

Crop and pad the background of the data to a size of **160×192×160** voxels.

- Cropping script: `preprocessing/2_cutbackground.py`
- Example cropped data: `preprocessing/cropdata/`

## Model Construction

### Model Overview

The proposed **CMTLNet** consists of two main components:

1. **CFE Module Training**
2. **Final CMTLNet Training**

Each component utilizes different data loaders.

- **Data Loaders**:
  - CFE Module Data Loader: `dataLoader/loader_S1.py`
  - CMTLNet Module Data Loader: `dataLoader/loader_S2.py`

- **Model Implementation**:
  - Full CMTLNet implementation: `model/CMTLNet.py`
  - **Note**: The final CMTLNet training should be conducted after the CFE module has successfully converged.

### Training Instructions

1. **Train the CFE Module**:
   - Execute the corresponding training script to ensure the CFE module converges.
   - Example training script: `Train_CMTLNet_Step1.py`
   - 
2. **Train the Final CMTLNet**:
   - After the CFE module has converged, run `Train_CMTLNet_Step2.py` to perform the final training.
   - Example training script: `Train_CMTLNet_Step2.py`

## File Structure
```markdown
CMTLNet
├── dataLoader
│   ├── dataloader_utils.py              # Utility functions for data loading
│   ├── loader_S1.py                     # Data loader for CFE Module
│   └── loader_S2.py                     # Data loader for CMTLNet
├── model
│   ├── CMTLNet.py                       # Full implementation of the CMTLNet model
│   ├── densenet.py                      # DenseNet backbone implementation
│   └── utils.py                         # Utility functions for the model
├── preprocessing
│   ├── cropdata/                        # Directory for cropped data
│   ├── regdata/                         # Directory for registration data
│   │   └── (spgr_unstrip_lps_b.nii.gz)  # Skull-stripped registration templates
│   ├── 1_registration.py                # Script for image registration
│   └── 2_cutbackground.py               # Script for cropping and padding background
├── Train_CMTLNet_Step1.py               # Training script for the CFE Module
├── Train_CMTLNet_Step2.py               # Training script for the final CMTLNet
├── config.py                            # Configuration file for training parameters
└── utils.py                             # General utility functions
```
## License
This project is licensed under the MIT License.

## Acknowledgement
Thanks to all the providers of the public datasets used in this project.
Appreciation to the contributors of the open-source projects such as FSL, MRIConvert and ANTsPy.

## Citation
If you use this code for your research, please cite our paper:
```bibtex
@article{chen2024103435,
  title = {Cooperative multi-task learning and interpretable image biomarkers for glioma grading and molecular subtyping},
  journal = {Medical Image Analysis},
  pages = {103435},
  year = {2024},
  issn = {1361-8415},
  author = {Qijian Chen and Lihui Wang and Zeyu Deng and Rongpin Wang and Li Wang and Caiqing Jian and Yue-Min Zhu}
}
```
You can also find the paper at the following DOI link:  
[https://doi.org/10.1016/j.media.2024.103435](https://doi.org/10.1016/j.media.2024.103435)
