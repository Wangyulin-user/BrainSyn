import models_ours
import torch
import SimpleITK as sitk
from scipy import ndimage as nd
import utils
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np
import os
from itertools import product
from ssim import SSIM
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from CLIP.model import CLIP
from utils_clip import load_config_file
import time
from concurrent.futures import ThreadPoolExecutor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint_path = '.../saved_checkpoints/checkpoint_42_42000.pt'
MODEL_CONFIG_PATH = './CLIP/model_config.yaml'
model_config = load_config_file(MODEL_CONFIG_PATH)

tokenizer = SimpleTokenizer()
model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModel

model_name_bert = ".../MedBERT/PubMedBERT"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name_bert)
model_bert = AutoModel.from_pretrained(model_name_bert).cuda()
model_bert.eval()

def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    original_spacing = img_ref.GetSpacing()
    original_size = img_ref.GetSize()
    resample = sitk.ResampleImageFilter()
    img = sitk.GetImageFromArray(vol)
    resample.SetOutputOrigin(img_ref.GetOrigin())
    resample.SetOutputDirection(img_ref.GetDirection())
    if new_spacing is None:
        resample.SetOutputSpacing(img_ref.GetSpacing())
    else:
        resample.SetOutputSpacing(tuple(new_spacing))
    size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    resample.SetSize(size)
    newimage = resample.Execute(img)
    sitk.WriteImage(newimage, out_path)
    
def set_new_spacing(ori_spacing, coord_size, crop_size):
    scale0 = coord_size[0] / crop_size[0]
    scale1 = coord_size[1] / crop_size[1]
    scale2 = coord_size[2] / crop_size[2]
    new_spacing = (ori_spacing[0]/scale0, ori_spacing[1]/scale1, ori_spacing[2]/scale2)
    return new_spacing

def tokenize(texts, tokenizer, context_length=116):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

def tokenize_bert(texts, tokenizer, context_length=116):
        if isinstance(texts, str):
            texts = [texts]
            
        sot_token = tokenizer_bert("<|startoftext|>")["input_ids"]
        eot_token = tokenizer_bert("<|endoftext|>")["input_ids"]
        all_tokens = [sot_token + tokenizer_bert(text)["input_ids"] + eot_token for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result
        
def img_pad(img, target_shape):
    current_shape = img.shape
    pads = [(0, max(0, target_shape[i] - current_shape[i])) for i in range(len(target_shape))]
    padded_img = np.pad(img, pads, mode='constant', constant_values=0)
    current_shape_2 = padded_img.shape
    crops = []
    for i in range(len(target_shape)):
        if current_shape_2[i] > target_shape[i]:
            crops.append(
                slice((current_shape_2[i] - target_shape[i]) // 2, (current_shape_2[i] + target_shape[i]) // 2))
        else:
            crops.append(slice(None))
    cropped_img = padded_img[tuple(crops)]
    return cropped_img


def calculate_patch_index(target_size, patch_size, overlap_ratio=0.25):
    shape = target_size
    gap0 = int(patch_size[0] * (1 - overlap_ratio))
    gap1 = int(patch_size[1] * (1 - overlap_ratio))
    gap2 = int(patch_size[2] * (1 - overlap_ratio))
    
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap0]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap1]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap2]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0] - patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1] - patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2] - patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos


def _get_pred(crop_size, overlap_ratio, model, img_vol_0, img_vol_1, seq_src, seq_tgt):
    W, H, D = img_vol_0.shape
    W_po, H_po, D_po = crop_size[0], crop_size[1], crop_size[2]
    pos = calculate_patch_index((W, H, D), crop_size, overlap_ratio)
    pred_0_1 = np.zeros((W, H, D))
    pred_1_0 = np.zeros((W, H, D))
    freq_rec = np.zeros((W, H, D))
    start_time = time.time()
    for start_pos in pos:
        img_0_lr_patch = img_vol_0[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1], start_pos[2]:start_pos[2] + crop_size[2]]
        img_1_lr_patch = img_vol_1[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1], start_pos[2]:start_pos[2] + crop_size[2]]
        img_0_lr_patch = torch.tensor(img_0_lr_patch).cuda().float().unsqueeze(0)
        img_1_lr_patch = torch.tensor(img_1_lr_patch).cuda().float().unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            pred_0_1_patch, pred_1_0_patch, _, _, _, _, _, _ = model(img_0_lr_patch, img_1_lr_patch, seq_src.unsqueeze(0).cuda().float(), seq_tgt.unsqueeze(0).cuda().float())

        pred_0_1_patch = pred_0_1_patch.squeeze(0).cpu().numpy()
        pred_1_0_patch = pred_1_0_patch.squeeze(0).cpu().numpy()

        pred_0_1[start_pos[0]:start_pos[0] + W_po, start_pos[1]:start_pos[1] + H_po, start_pos[2]:start_pos[2] + D_po] += pred_0_1_patch[:, :, :]
        pred_1_0[start_pos[0]:start_pos[0] + W_po, start_pos[1]:start_pos[1] + H_po, start_pos[2]:start_pos[2] + D_po] += pred_1_0_patch[:, :, :]
        freq_rec[start_pos[0]:start_pos[0] + W_po, start_pos[1]:start_pos[1] + H_po, start_pos[2]:start_pos[2] + D_po] += 1
    end_time = time.time()
    print(end_time-start_time)
    pred_0_1_img = pred_0_1 / freq_rec
    pred_1_0_img = pred_1_0 / freq_rec

    return pred_0_1_img, pred_1_0_img

def psnr(ref,ret):
    err = ref - ret
    return -10*np.log10(np.mean(err**2))

def mae(ref,ret):
    mae = np.mean(np.abs(ref - ret))
    return mae

psnr_0_1_list = []
psnr_1_0_list = []
ssim_0_1_list = []
ssim_1_0_list = []
mae_0_1_list = []
mae_1_0_list = []
model_pth = '.../save/_train_lccd_sr/epoch-100.pth'
model_img = models_ours.make(torch.load(model_pth)['model_G'], load_sd=True).cuda()
img_path_0 = #image root (modality 1)
img_path_1 = #image root (modality 2)

img_list_0 = sorted(os.listdir(img_path_0))
img_list_1 = sorted(os.listdir(img_path_1))

prompt_M1 = #prompt root (modality 1)
prompt_M2 = #prompt root (modality 2)
with open(prompt_M1) as f1, open(prompt_M2) as f2:
    lines_M1 = f1.readlines()
    lines_M2 = f2.readlines()

for idx, (i, j) in enumerate(zip(img_list_0, img_list_1)):
    start_time_0 = time.time()
    img_0 = sitk.ReadImage(os.path.join(img_path_0, i))
    img_0_spacing = img_0.GetSpacing()
    img_vol_0 = sitk.GetArrayFromImage(img_0)
    W, H, D = img_vol_0.shape
    img_1 = sitk.ReadImage(os.path.join(img_path_1, j))
    img_1_spacing = img_1.GetSpacing()
    img_vol_1 = sitk.GetArrayFromImage(img_1)
    img_vol_0 = utils.percentile_clip(img_vol_0)
    img_vol_1 = utils.percentile_clip(img_vol_1)
    text_src = lines_M1[idx].replace('"', '')
    text_src = text_src.strip((text_src.strip().split(':'))[0])
    text_src = text_src.strip(text_src[0])
    text_tgt = lines_M2[idx].replace('"', '')
    text_tgt = text_tgt.strip((text_tgt.strip().split(':'))[0])
    text_tgt = text_tgt.strip(text_tgt[0])
    seq_src = tokenize(text_src, tokenizer).cuda()
    with torch.no_grad():
        seq_src = model.encode_text(seq_src)
    seq_tgt = tokenize(text_tgt, tokenizer).cuda()
    with torch.no_grad():
        seq_tgt = model.encode_text(seq_tgt)

    seq_src_bert = tokenize_bert(text_src, tokenizer_bert).cuda()
    with torch.no_grad():
        seq_src_bert = model_bert(seq_src_bert)
    src_bert_embedding = seq_src_bert.last_hidden_state[:, 0, :]
    seq_tgt_bert = tokenize_bert(text_tgt, tokenizer_bert).cuda()
    with torch.no_grad():
        seq_tgt_bert = model_bert(seq_tgt_bert)
    tgt_bert_embedding = seq_tgt_bert.last_hidden_state[:, 0, :]
    
    seq_src = torch.cat((seq_src, src_bert_embedding),dim=1)
    seq_tgt = torch.cat((seq_tgt, tgt_bert_embedding),dim=1)

    crop_size = (4, H, D)
    pred_0_1, pred_1_0 = _get_pred(crop_size, 0.5, model_img, img_vol_0, img_vol_1, seq_src, seq_tgt)
    new_spacing_0 = img_0_spacing
    new_spacing_1 = img_1_spacing
    
    utils.write_img(pred_0_1, os.path.join('', 'M1_M2_'+i.strip().split('/')[-1]), j.strip(), new_spacing=new_spacing_1)
    utils.write_img(pred_1_0, os.path.join('', 'M2_M1_'+i.strip().split('/')[-1]), i.strip(), new_spacing=new_spacing_0)