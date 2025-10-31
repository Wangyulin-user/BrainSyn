import random
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from datasets import register
import numpy as np
from scipy import ndimage as nd
import utils


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, scale_min=1, scale_max=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        patch_src, patch_tgt, seq_src, seq_tgt, srcroot, tgtroot = self.dataset[idx]
        
        non_zero_src = np.nonzero(patch_src)
        non_zero_tgt = np.nonzero(patch_tgt)
        min_indice_src = np.min(non_zero_src, axis=1)
        max_indice_src = np.max(non_zero_src, axis=1)
        min_indice_tgt = np.min(non_zero_tgt, axis=1)
        max_indice_tgt = np.max(non_zero_tgt, axis=1)
        if min_indice_src[0] < min_indice_tgt[0]:
            patch_src = patch_src[min_indice_tgt[0]:max_indice_tgt[0]+1, min_indice_tgt[1]:max_indice_tgt[1]+1, min_indice_tgt[2]:max_indice_tgt[2]+1]
            patch_tgt = patch_tgt[min_indice_tgt[0]:max_indice_tgt[0]+1, min_indice_tgt[1]:max_indice_tgt[1]+1, min_indice_tgt[2]:max_indice_tgt[2]+1]
        else: 
            patch_src = patch_src[min_indice_src[0]:max_indice_src[0]+1, min_indice_src[1]:max_indice_src[1]+1, min_indice_src[2]:max_indice_src[2]+1]
            patch_tgt = patch_tgt[min_indice_src[0]:max_indice_src[0]+1, min_indice_src[1]:max_indice_src[1]+1, min_indice_src[2]:max_indice_src[2]+1]
            
            
        size = 64
        
        if patch_src.shape != patch_tgt.shape:
            print(f"Skipping data at {srcroot} and {tgtroot} due to incompatible shape: {patch_src.shape} vs {patch_tgt.shape}")
            return self.__getitem__((idx + 1) % len(self.dataset)) 
        if any(dim < size for dim in patch_src.shape) or any(dim < size for dim in patch_tgt.shape) :
            print(f"Skipping {srcroot} with shape {patch_src.shape} as one dimension is smaller than {size}.")
            return self.__getitem__((idx + 1) % len(self.dataset)) 
        patch_src = utils.pad_img(patch_src, size)
        patch_tgt = utils.pad_img(patch_tgt, size)
        
        h_size = 4
        h0 = random.randint(0, patch_src.shape[0] - h_size)
        w0 = random.randint(0, patch_src.shape[1] - size)
        d0 = random.randint(0, patch_src.shape[2] - size)

        patch_src = patch_src[h0:h0 + h_size, w0:w0 + size, d0:d0 + size]
        patch_tgt = patch_tgt[h0:h0 + h_size, w0:w0 + size, d0:d0 + size]
        if np.isnan(patch_src).any() or np.isinf(patch_src).any() or np.isnan(patch_tgt).any() or np.isinf(patch_tgt).any() or (patch_src == 0).all() or (patch_tgt == 0).all():
            print(f"Skipping data at {srcroot} and {tgtroot} due to nan")
            return self.__getitem__((idx + 1) % len(self.dataset)) 


        return {
            'src': patch_src,
            'tgt': patch_tgt,
            'seq_src': seq_src.detach(),
            'seq_tgt': seq_tgt.detach()
        }


