# VasCab
Code and dataset of our paper "Collect Vascular Specimens in One Cabinet: A Hierarchical Prompt-Guided Universal Model for 3D Vascular Segmentation".

## Introduction
We present VasBench, a new comprehensive vascular segmentation benchmark comprising nine sub-datasets spanning diverse modalities and anatomical regions. Building on this foundation, we introduce \textbf{VasCab}, a novel prompt-guided universal model for volumetric vascular segmentation, designed to ``collect vascular specimens in one cabinet". Specifically, VasCab is equipped with learnable domain and topology prompts to capture shared and unique vascular characteristics across diverse data domains, complemented by morphology perceptual loss to address complex morphological variations. Experimental results demonstrate that VasCab surpasses individual models and state-of-the-art medical foundation models across all test datasets, showcasing exceptional cross-domain integration and precise modeling of vascular morphological variations.
<p align="center"><img width="80%" src="fig/figure_1_600.png" /></p>

## Updates
- 2025.05.24: Code released.

## Usage
### 1. Installation
```bash
$ git clone https://github.com/mileswyn/VasCab.git
$ cd VasCab/
$ pip install requirements.txt
```
For more details about Installation, please refer to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

### 2. Data
### Dataset Link
#### Baidu Cloud Drive
```
https://pan.baidu.com/s/1-F_2Uv0GasZBaKRCQIHhBQ (Extract Code: gwqc)
```

### 3. Code
#### step 1: dataset conversations
Use codes in /nnunet/dataset/conversation to handle each sub-dataset.
#### step 2: create plans
Run codes in /nnunet/experiment_planning to preprocess each sub-dataset.
#### step 3: Merge plans
Run /toolbox/merge_each_sub_datasets.py to merge and generate a new plan for training the universal model.
#### step 4: train models
Run /nnunet/run/run_training_universal.py
#### step 5: inference
Run /nnunet/inference/predict_simple_305_V3.py to infer each dataset. Remember to adapt the configuration.

## TODO
- [] Downstream code
- [] DPAM updated version
- [] Modular modification

## Citation

## Acknowlegment
We thank original dataset owners and the researchers of nnU-Net.
