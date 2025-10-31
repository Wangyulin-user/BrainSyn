# Towards a Foundation Model for Text-guided Brain Medical Image Synthesis



Welcome! This is the official implementation of BrainSyn, a generative foundation model that can synthesize high-quality brain images covering a wide spectrum of imaging modalities, guided by textual imaging parameters.
BrainSyn demonstrates unprecedented flexibility in performing both inter-modality synthesis (e.g., MRI-to-PET, CT-to-PET, and CT-to-MRI) and fine-grained intra-modality synthesis (e.g., 3T-to-7T MRI, FDG-PET-to-AV45-PET, and multi-cohort harmonization).

## Content

  - [CLIP](https://github.com/Wangyulin-user/BrainSyn/tree/main/CLIP): This folder contains the model architectures and hyper-parameter configs of image encoder and text encoder in stage 1.
  - [utils_clip](https://github.com/Wangyulin-user/BrainSyn/tree/main/utils_clip): This folder contains the utility functions required to run the models in stage 1.
  - [datasets](https://github.com/Wangyulin-user/BrainSyn/tree/main/datasets): This folder contains codes for the image synthesis model to process text and image data.
  - [configs](https://github.com/Wangyulin-user/BrainSyn/tree/main/configs): This folder contains `train_lccd_sr.yaml`, which is used to set all training/validation parameters.
  - [models](https://github.com/Wangyulin-user/BrainSyn/tree/main/models_ours): This folder contains `lccd_bi.py`, which is a wrapper around the model. It also contains the network architectures of all the modules used in the image synthesis model.
