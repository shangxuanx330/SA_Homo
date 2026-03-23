# SA-Homo

This repository provides the official implementation of SA-Homo, including the inference scripts, pre-trained model weights, and the benchmark datasets used. The training code will be relased later.

## 📂 Datasets


All datasets used in this paper are available at the following Hugging Face repository:
👉 **[SA_Homo Dataset Download](https://huggingface.co/datasets/ckkkk333000/SA_Homo)**

After downloading the validation datasets, please put them in the `datasets` folder like:

```text
datasets/
├── coco/
├── gfnet_dronevehicle/
├── GoogleEarth/
├── GoogleMap/
├── hmsa_1152x1152/
└── rgb_nir/
```
Detailed sources of the original datasets are as follows:

1. **HMSA**
   - This is the dataset contributed by our paper，the **validation part** is provided now.
     
2. **MSCOCO (2014)**
   - The complete MSCOCO 2014 can be download from Official Website: [https://cocodataset.org/#download](https://cocodataset.org/#download).

3. **GoogleMap & GoogleEarth**
   - GoogleMap and GoogleEarth can be download from source repository: [CVPR21-Deep-Lucas-Kanade-Homography](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography).
   
4. **DroneVehicle**
   - Original dataset source: [VisDrone/DroneVehicle](https://github.com/VisDrone/DroneVehicle)
   - This paper utilizes the processed version produced by GFNet [GFNet (KN-Zhang)](https://github.com/KN-Zhang/GFNet).
     
5. **RGB-NIR**
   - The complete RGB-NIR multimodal scene dataset can be download form repository: [Multimodal_Feature_Evaluation](https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation).
     
---
*If you use these datasets, please adhere to the license agreements of the original dataset authors and cite the corresponding original literature in your papers.*


## 📂 Pre-trained Models

The main pre-trained models are available at the following Hugging Face repository:
👉 **[SA_Homo Dataset Download](https://huggingface.co/ckkkk333000/SA_Homo)**

if you do not want to change the inference scripts provided, please place the downloaded checkpoint files into the `ckpts` folder according to the following structure:
```text
ckpts/
├── coco/
├── dronevehicle/
├── googleearth/
├── googlemap/
├── hmsa/
└── rgb_nir/
