# GlucoSense: Non-Invasive Blood Glucose Sensing on Mobile Devices  
This repository describes the detailed steps to reproduce the research results presented in the paper titled: 
[``GlucoSense: Non-Invasive Blood Glucose Sensing on Mobile Devices``](https://www.cs.sfu.ca/~mhefeeda/Papers/mobicom25_GlucoSense.pdf) published in ACM MobiCom'25. 

*[Neha Sharma](https://www.linkedin.com/in/sharma-neha512/)*<sup>1</sup>, *[Mariam Bebawy](https://www.linkedin.com/in/mariam-bebawy/)*<sup>1</sup>, *[Yik Yu Ng](https://www.linkedin.com/in/yik-yu-ng-7847b62a7/)*<sup>2</sup>, and *[Mohamed Hefeeda](https://www.linkedin.com/in/mohamed-hefeeda-71a5445/)*<sup>1,3</sup>  

<span style="font-size:9px"><sup>1</sup> School of Computing Science, Simon Fraser University, Canada</span>

<span style="font-size:9px"><sup>2</sup> School of Computer Science, McGill University, Canada</span>

<span style="font-size:9px"><sup>3</sup> Qatar Computing Research Institute, Qatar</span>

## 💡 Overview

GlucoSense enables regular blood glucose monitoring using only smartphones, leveraging mobile RGB and NIR cameras, hyperspectral reconstruction, and machine learning models. 
It is designed to work with unmodified phones and achieves clinically acceptable accuracy as demonstrated by comparison with an FDA-approved glucose monitoring device in a user study.

### Core Components

- **Mobile Sensing:** Leverages the standard RGB camera and the near-infrared (NIR) sensors (e.g., depth-sensing cameras like Time-of-Flight) available on modern smartphones.  
- **HyperSpectral Reconstruction:** A deep learning model converts the sparse RGB/NIR inputs into 50 crucial spectral bands (400–1000 nm), focusing on wavelengths proven most important for glucose detection (S4.2 in the paper).  
- **Glucose Estimation:** An XGBoost regression model maps the reconstructed spectral bands to the final blood glucose level (mg/dL).  

### Key Results

In an ethics-approved user study comparing GlucoSense against an FDA-approved CGM device (FreeStyle Libre 2), GlucoSense achieved the following clinical accuracy (RGB+NIR system):

- **Clarke Error Grid (CEG):** 80.4% in Zone A (Clinically Accurate) and 19.3% in Zone B (Clinically Acceptable).  
- **Surveillance Error Grid (SEG):** 99.7% of predictions were within the None and Slight risk zones.  



## 💻 Installation Instructions

### Prerequisites
- Workstation running **Linux**
- **NVIDIA GPU + CUDA CuDNN** (required for training and inference with the reconstruction model)
- Python Anaconda (**Python version 3.8** is recommended)

### Install the code to an new environment  

1. Download miniconda (Linux Python 3.8) from the official website and install it: https://docs.conda.io/en/latest/miniconda.html  
```bash
bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
```

2. Clone the repository, create the Anaconda environment, and install dependencies:
```bash
git clone [https://github.com/mariam-bebawy/glucosense-mobicom25.git](https://github.com/mariam-bebawy/glucosense-mobicom25.git)
cd glucosense-mobicom25
conda create --name glucosense python=3.8
conda activate glucosense

# Install Pytorch and other dependencies
# Ensure you match the PyTorch version to your CUDA version if training or using GPU
pip install torch==1.8.1+cu111 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
pip install -r requirements.txt
```

### Repository Structure

```bash
glucosense/
├── datasets/             # Holds the GlucoSense User Study Dataset (after download)
├── reconstruction/       # Code for the HyperSpectral Reconstruction Model
│   ├── train/
│   ├── test/             # Test reconstruction accuracy against ground truth HSI
│   └── evaluate_mobile/  # Apply HSI reconstruction to mobile camera data (RGB+NIR, etc.)
├── regression/           # Code for the Glucose Estimation Model (XGBoost, SVR, etc.)
│   ├── Architecture/
│   └── ... 
├── pretrained_models/    # Location for pre-trained models (Reconstruction & Regression)
├── LICSENSE
└── README.md
```

## 🧪 Training and Testing Reconstruction (Optional)  

This section details how to train the HyperSpectral Reconstruction model from scratch and verify its accuracy against the HSI ground truth.  

### 1. Training the Reconstruction Model from Scratch  

To train the MST++ model on the Human Study Hyperspectral Image (HSI) dataset (as detailed in SA.1 of the paper), run the following command.

Note: This process is computationally intensive and may take several hours depending on your GPU.

```bash
cd reconstruction/train
# This trains the model on the HSI data for the 23 participants used in the training split.

python3 train.py --method mst_plus_plus \
    --data_root ../../../datasets/HSDatasets/ \
    --outf ./exp/mst_glucosense_run/ \
    --batch_size 64 \
    --end_epoch 100 \
    --init_lr 4e-4 \
    --patch_size 64 \
    --stride 64 \
    --gpu_id 0
```  
*The resulting model checkpoint will be saved in the specified output directory (./exp/mst_glucosense_run/).*  

### 2. Testing Reconstruction Accuracy  

To test the accuracy of a trained model against the HSI ground truth data (Table 3 in the paper) on the 8-participant test split:

```bash
cd reconstruction/test
# Use either your newly trained model or the provided pre-trained model path.

python3 train.py --method mst_plus_plus \
    --data_root ../../../datasets/HSDatasets/ \
    --method mst_plus_plus \
    --pretrained_model_path ../evaluation_mobile/Models/mst_AWB_940_t50.pth \
    --outf ../../../datasets/HSDatasets/rec_50/  \
    --gpu_id 0
```  
*The performance metrics (RMSE, SAM, SID, PSNR) will be printed to the console.*  


## 💾 Reproducing Results  

### 1. Download Datasets  

- **GlucoSense User Study Dataset:** The necessary data for reproduction—consisting of hyperspectral (HSI) images for reconstruction model training and paired mobile images from the 31 participants for validation across the four hardware systems—is required and available as an open-source data package [here](https://1sfu-my.sharepoint.com/:f:/g/personal/mba216_sfu_ca/Es8O3o3jPJRIi5OqAWj9RvAB09bnrEoC5fG7hKq5tYWBqA?e=tKHyBK).

- Unzip the downloaded dataset and place the contents (e.g., `HSDatasets/` and `MobileDatasets/` splits) into the `datasets/` folder.

### 2. Download Pre-trained Models

You will need the pre-trained models for the reconstruction and glucose estimation steps.

- **Pre-trained Reconstruction Model:** Download the model files (e.g., `mst_AWB_940_t50.pth`) [here](https://1sfu-my.sharepoint.com/:f:/r/personal/mba216_sfu_ca/Documents/GlucoSense_MobiCom25/Pretrained%20Models/Reconstruction?csf=1&web=1&e=JObLaU). Place it in the `reconstruction/Models/` folder.

- **Pre-trained Glucose Estimation Models:** The trained XGBoost regression models for each imaging system (e.g., `xgboost_onsemi.pkl`) should be downloaded from [here](https://1sfu-my.sharepoint.com/:f:/r/personal/mba216_sfu_ca/Documents/GlucoSense_MobiCom25/Pretrained%20Models/Regression?csf=1&web=1&e=xdbAu3) and placed in the `regression/Models/` folder.

### 3. Execution Steps (RGB+NIR System)

The following sequence of commands demonstrates how to use the pre-trained models to predict glucose levels using the mobile image test set, focusing on the RGB+NIR system (the one corresponding to the unmodified Google Pixel phone).

#### A. Apply HyperSpectral Reconstruction to Mobile Data  

This step simulates the reconstruction of the 50 spectral bands from the sparse RGB/NIR test images.  

```bash
cd reconstruction/evaluate_mobile
# Assuming the human study mobile images are organized under datasets/human_study/
# The output folder will contain the reconstructed spectral bands (.mat files).

python3 test.py --data_root ../../datasets/human_study/mobile_data/RGB+NIR/ \
    --method mst_plus_plus \
    --pretrained_model_path ../../pretrained_models/mst_AWB_940_t50.pth \
    --outf ../../datasets/human_study/reconstructed_bands/RGB+NIR/  \
    --gpu_id 0
```

#### B. Predict Glucose Levels

This step feeds the reconstructed bands into the glucose estimation model and runs the full evaluation against the reference CGM data.

```bash
cd regression/Architecture
# This script loads the reconstructed bands and the XGBoost model to output the CEG/SEG analysis.

python3 main.py
```
*The output metrics will include the MARD, and the zone percentages for the Clarke and Consensus/Surveillance Error Grids, matching the results in $\S$6.4 of the paper.*  

## 📝 Citation

If you use our code or dataset for your research, please cite our paper:

```
@inproceedings{10.1145/3680207.3723472,
author = {Sharma, Neha and Bebawy, Mariam and Ng, Yik Yu and Hefeeda, Mohamed},
title = {GlucoSense: Non-Invasive Glucose Monitoring using Mobile Devices},
year = {2025},
isbn = {},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {[https://doi.org/10.1145/3680207.3723472](https://doi.org/10.1145/3680207.3723472)},
doi = {10.1145/3680207.3723472},
booktitle = {Proceedings of the 31st Annual International Conference on Mobile Computing and Networking},
articleno = {},
numpages = {},
keywords = {blood glucose, mobile health, hyperspectral imaging},
location = {Hong Kong, China},
series = {ACM MobiCom '25}
}
```  