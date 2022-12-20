### Program to sample from the trained unclip model based on input prompts

import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm as tqdm
import sentencepiece as spm

from decoder import * # this internally imports everything needed


# utility function to tokenize prompt and obtain masks
def process_prompt(prompt, spm_processor, sos_token, eos_token, pad_token, max_seq_len, device):
    tokenized_captions = []
    caption_tokens = spm_processor.encode(prompt, out_type=int)
    caption_tokens = [sos_token] + caption_tokens + [eos_token] # append sos and eos tokens
    while len(caption_tokens) < max_seq_len:
        caption_tokens.append(pad_token) # padding
    tokenized_captions.append(caption_tokens)
    tokenized_captions = torch.LongTensor(tokenized_captions).to(device)
    # get caption mask
    cap_mask_padding = pad_mask(tokenized_captions, pad_token) # pad mask for captions
    cap_mask_causal = subsequent_mask(tokenized_captions.shape).to(device)
    return tokenized_captions, cap_mask_padding, cap_mask_causal


def save_img(img, name):
    name = 'prompt_samples/' + name
    img = (img.clamp(-1, 1) + 1) * 0.5
    img = (img * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(img)
    ndarr = grid.permute(2, 1, 0).to('cpu').numpy()
    cv2.imwrite(name, ndarr)


### main - to setup and train Decoder (Glide parameterized by Unet)
if __name__ == '__main__':

    # hyperparams for transformer
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_len_clip = 53 # based on the dataset_coco_val2017 that the model was trained on

    # hyperparams for diffusion
    L_lambda = 0.001 # for weighing L_vlb
    guidance_strength = 3 # w in classifier free guidance paper
    p_uncond = 0.1 # probability for setting conditioning_embedding = None
    beta_min = 1e-4 # not needed for cosine noise schedule
    beta_max = 0.02 # not needed for cosine noise schedule
    max_time_steps = 1000 # 4000
    subseq_steps = 200 # used for both strided sampling and ddim sampling - should be less than max_time_steps

    img_size_resnet = 224 # resize for resnet
    img_size_unet = 64 # resize for unet
    img_unet_shape = [3, 64, 64]
    clip_img_emb_shape = [d_model]
    lr = 3e-4
    max_epochs = 100
    random_seed = 1010

    ckpt_path_clip = 'ckpts_clip_resnet/latest.pt'
    ckpt_path_dpct = 'ckpts_dpct/latest.pt'
    ckpt_path_glide = 'ckpts_decoder/latest.pt'

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # clip_device = torch.device('cpu') # clip model will be loaded for inference on the cpu (to save memory on the gpu for diffusion prior model)
    clip_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dpct_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load spm as tokenizer
    spm_processor = spm.SentencePieceProcessor(model_file='spm1.model')
    vocab_size_clip = len(spm_processor)

    # declare sos, eos, pad tokens
    sos_token, eos_token, unk_token = spm_processor.piece_to_id(['<s>', '</s>', '<unk>'])
    pad_token = unk_token # <unk> token is used as the <pad> token

    # init clip model
    clip_image_encoder = ImageEmbeddings(forward_hook, d_model) # resnet
    clip_text_encoder = init_text_encoder(vocab_size_clip, max_seq_len_clip, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, clip_device)
    clip_model = init_CLIP(clip_text_encoder, clip_image_encoder, d_model, eos_token).to(clip_device)
    # load clip_model weights and put clip_model in eval mode
    clip_model = load_ckpt(ckpt_path_clip, clip_model, device=clip_device, mode='eval')

    # init DPCT
    max_seq_len_dpct = max_seq_len_clip + 4 # for clip_txt_emb, time_step, clip_img_emb, final_emb
    dpct_model = init_dpct(max_seq_len_dpct, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(dpct_device)
    # load dpct_model weights and put dpct in eval mode
    dpct_model = load_ckpt(ckpt_path_dpct, dpct_model, device=dpct_device, mode='eval')

    # init decoder (Glide parameterized by UNet with modifications on UNet for classifier-free guidance)
    decoder = init_unclip_decoder(d_model, device).to(device)
    # load decoder weights and put decoder in eval mode
    decoder = load_ckpt(ckpt_path_glide, decoder, device=device, mode='eval')

    # calcualate betas and alphas
    betas = linear_noise_schedule(beta_min, beta_max, max_time_steps)
    # betas = cosine_noise_schedule(max_time_steps)
    # betas = betas_for_alpha_bar(max_time_steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alphas_hat = alphas_hat
    alphas_hat = alphas_hat.to(device) # alphas_hat.shape: [max_time_steps]

    # optimizer and loss criterion - not needed when we are only sampling
    # optimizer = torch.optim.AdamW(params=decoder.parameters(), lr=lr)

    # sample
    for ep in tqdm(range(max_epochs)):

        # set prompt
        prompt = input('Enter prompt: ')

        # tokenize prompt and obtain masks
        cap_tokens, cap_mask_padding, cap_mask_causal = process_prompt(prompt, spm_processor, sos_token, eos_token, pad_token, max_seq_len_clip, device)

        with torch.no_grad():
            # encode (inference) through CLIP
            # clip_img_emb.shape: [batch_size, d_model]
            # clip_txt_emb.shape: [batch_size, d_model]
            # clip_encoded_txt.shape: [batch_size, max_seq_len_clip, d_model]
            clip_txt_emb, clip_encoded_txt = clip_model.encode_text(cap_tokens, cap_mask_padding, cap_mask_causal)

            # move from clip_device to dpct_device
            clip_txt_emb, clip_encoded_txt = clip_txt_emb.to(dpct_device), clip_encoded_txt.to(dpct_device)

            # concatenating clip_encoded_txt and clip_txt_emb for interfacing with other functions
            clip_txt_emb = clip_txt_emb.unsqueeze(1) # clip_txt_emb.shape: [batch_size, 1, d_model]
            clip_txt_out = torch.cat((clip_encoded_txt, clip_txt_emb), dim=1) # clip_txt_out.shape: [batch_size, max_seq_len_clip + 1, d_model]

            # get sampled_clip_img_emb from the diffusion_prior
            sampled_clip_img_emb = sample_strided_CFG(dpct_model, alphas_hat, guidance_strength, max_time_steps, subseq_steps, clip_img_emb_shape, device, 1, clip_txt_out, cfg_flag=False, pred_x_start=True, progress_bar=False) # sampled_clip_img_emb.shape: [batch_size, d_model]

            # move from dpct_device to (decoder) device
            sampled_clip_img_emb = sampled_clip_img_emb.to(device)

            sampled_img = sample_strided_CFG(decoder, alphas_hat, guidance_strength, max_time_steps, subseq_steps, img_unet_shape, device, 1, sampled_clip_img_emb, cfg_flag=True)
            # save sampled_img
            save_img(sampled_img, prompt + '.png')
