import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from datasets import register
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np
from CLIP.model import CLIP
from utils_clip import load_config_file


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path_1, root_path_2, prompt_D1_M1, prompt_D1_M2, repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.root_path_1 = root_path_1
        self.root_path_2 = root_path_2
        self.prompt_D1_M1 = prompt_D1_M1
        self.prompt_D1_M2 = prompt_D1_M2

        with open(self.root_path_1) as f1, open(self.root_path_2) as f2, open(self.prompt_D1_M1) as f3, open(
                self.prompt_D1_M2) as f4:
            img_M1 = f1.readlines()
            img_M2 = f2.readlines()
            prompt_M1 = f3.readlines()
            prompt_M2 = f4.readlines()
        self.img_M1 = img_M1
        self.img_M2 = img_M2
        self.prompt_M1 = prompt_M1
        self.prompt_M2 = prompt_M2

    def __len__(self):
        return len(self.img_M1) * self.repeat

    def __getitem__(self, idx):
        patch_src_hr = self.img_M1[idx % len(self.img_M1)]
        patch_tgt_hr = self.img_M2[idx % len(self.img_M1)]

        text_src = self.prompt_M1[idx % len(self.img_M1)]
        text_tgt = self.prompt_M2[idx % len(self.img_M1)]
        
        seq_src = torch.load(text_src.strip())
        seq_tgt = torch.load(text_tgt.strip())

        if patch_src_hr.strip().endswith('.npy'):
            img_vol_src_hr = np.load(patch_src_hr.strip())  
            img_vol_tgt_hr = np.load(patch_tgt_hr.strip())
        else:
            img_vol_src_hr = sitk.GetArrayFromImage(sitk.ReadImage(patch_src_hr.strip()))
            img_vol_tgt_hr = sitk.GetArrayFromImage(sitk.ReadImage(patch_tgt_hr.strip()))

        return img_vol_src_hr, img_vol_tgt_hr, seq_src, seq_tgt, patch_src_hr.strip(), patch_tgt_hr.strip()  # last_hidden_state


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, prompt_D1_M1, prompt_D1_M2, repeat, cache, **kwargs):
        self.dataset = ImageFolder(root_path_1, root_path_2, prompt_D1_M1, prompt_D1_M2, repeat, cache, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]