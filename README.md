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

### 2. Data
### Dataset Link
#### Baidu Cloud Drive
```
https://pan.baidu.com/s/1TlxqZHl6T17on_f3BDixmA (Extract code：hwwd)
```
#### Google Cloud Drive
```
https://drive.google.com/file/d/1idT6rCKXry-Yc8GF-DBxEn6e32-pBiFF/view?usp=sharing
```
