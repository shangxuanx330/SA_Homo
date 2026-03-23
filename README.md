# SA-Homo: Scale Adaptive Homography Estimation for Scale Variation Scenarios

The inference code, pre-trained models, and validation datasets are currently being organized and continuously uploaded. All uploads are expected to be completed within one week.



## 📂 Datasets


All datasets used in this project are available at the following Hugging Face repository:
👉 **[SA_Homo Dataset Download](https://huggingface.co/datasets/ckkkk333000/SA_Homo)**

2. Directory Structure
After downloading and extracting the datasets, please arrange them in the `datasets` folder according to the following structure to ensure the code runs correctly:

```text
datasets/
├── coco/
├── gfnet_dronevehicle/
├── GoogleEarth/
├── GoogleMap/
├── hmsa_1152x1152/
└── rgb_nir/

Detailed sources of the original datasets are as follows:

1. **HMSA**
   - This is the dataset contributed by our paper. 
   - **Note**: The **validation part** is provided here. The **training part** will be released in subsequent updates.
     
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
