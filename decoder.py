### Program implementing the decoder for unclip (glide)

## Features:
# 1. The decoder is glide, parameterized by a Unet, predicts the noise, training loss function is mse over pred_noise and true_noise with VLB loss for variance.
# 2. The decoder uses clip image embeddings for conditioning (trained in classifier-free guidance fashion)
# 3. This implementation of decoder does not use the glide text encoder to condition the Unet on caption_encodings (along with clip_img_emb) - as the paper suggests its of little help.

## Todos / Questions:
# 1.

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

from diffusion_prior import * # note that this import other utils as well (utils_transformer, utils_diffusion, and functions from clip_resnet)
from utils_unet import * # import UNet_Glide for parameterizing the decoder (glide model)

# class implementing the decoder (which is a UNet with some handling)
class Unclip_Decoder(nn.Module):
    def __init__(self, unet, d_model):
        super().__init__()
        self.net = unet
        self.d_model = d_model
        self.clip_img_emb_proj = nn.Linear(d_model, 4 * d_model, bias=False) # 4 extra tokens of context as mentioned in dalle2 paper section 2.1
    def forward(self, x_t, t, clip_img_emb): # clip_img_emb.shape: [batch_size, d_model]
        batch_size = x_t.shape[0]
        if clip_img_emb is not None:
            clip_img_emb_context = self.clip_img_emb_proj(clip_img_emb)  # clip_img_emb_context.shape: [batch_size, 4 * d_model]
            clip_img_emb_context = clip_img_emb_context.view(batch_size, 4, self.d_model) # clip_img_emb_context.shape: [batch_size, 4, d_model]
            out = self.net(x_t, t, clip_img_emb, clip_img_emb_context) # out.shape: [batch_size, img_channels * 2 = 6, d_model]
        else:
            out = self.net(x_t, t, None, None) # out.shape: [batch_size, img_channels * 2 = 6, d_model]
        return out

# caller function to init decoder
def init_unclip_decoder(d_model, device):
    unet = UNet_Glide(c_in=3, c_out=6, caption_emb_dim=d_model, device=device)
    model = Unclip_Decoder(unet, d_model)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# utility function to preprocess minibatch images for unet (we want separate image size for unet than for resnet)
def process_batch_images_for_unet(minibatch, img_size, device):
    augmented_imgs = []
    imgs_folder = 'dataset_coco_val2017/images/'
    for img_filename, caption_text in minibatch:
        # obtain augmented img from img_filename
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.transpose(1,2).transpose(0,1).transpose(1,2) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
        ])
        img = transforms(img)
        augmented_imgs.append(img)
    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    return augmented_imgs


def save_img_CFG(img, name):
    name = 'generated_decoder/' + name
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
    lr = 3e-4
    batch_size = 2
    max_epochs = int(5644 * 17)
    random_seed = 1010

    ckpt_path_clip = 'ckpts_clip_resnet/latest.pt'
    ckpt_path_dpct = 'ckpts_dpct/latest.pt'

    ckpt_path_glide = 'ckpts_decoder/latest.pt'
    resume_training_from_ckpt = True

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

    # load data and create img_cap_dict
    img_cap_dict, max_seq_len_clip = load_data()

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

    # calcualate betas and alphas
    betas = linear_noise_schedule(beta_min, beta_max, max_time_steps)
    # betas = cosine_noise_schedule(max_time_steps)
    # betas = betas_for_alpha_bar(max_time_steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alphas_hat = alphas_hat
    alphas_hat = alphas_hat.to(device) # alphas_hat.shape: [max_time_steps]

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=decoder.parameters(), lr=lr)

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        decoder, optimizer = load_ckpt(ckpt_path_glide, decoder, optimizer, device=device, mode='train')

    # train
    for ep in tqdm(range(max_epochs)):

        # fetch minibatch - a batch of [img_filename, caption_text] pairs
        minibatch_keys = np.random.choice(list(img_cap_dict.keys()), size=batch_size)
        minibatch = [[k, img_cap_dict[k]] for k in minibatch_keys]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        # note that we don't create embeddings yet, since that's done by the image_encoder and text_encoder in the CLIP model
        imgs_resnet, cap_tokens, cap_mask_padding, cap_mask_causal = process_batch(minibatch, clip_model, spm_processor, sos_token, eos_token, pad_token, max_seq_len_clip, img_size_resnet, clip_device) # imgs_resnet.shape:[batch_size, 3, 224, 224], captions.shape:[batch_size, max_seq_len_clip]

        imgs_unet = process_batch_images_for_unet(minibatch, img_size_unet, device) # imgs_unet.shape:[batch_size, 3, 64, 64]

        with torch.no_grad():
            # encode (inference) through CLIP
            # clip_img_emb.shape: [batch_size, d_model]
            # clip_txt_emb.shape: [batch_size, d_model]
            # clip_encoded_txt.shape: [batch_size, max_seq_len_clip, d_model]
            clip_img_emb, clip_txt_emb, clip_encoded_txt = clip_model.encode(imgs_resnet, cap_tokens, cap_mask_padding, cap_mask_causal)

            # move from clip_device to dpct_device
            clip_img_emb, clip_txt_emb, clip_encoded_txt = clip_img_emb.to(dpct_device), clip_txt_emb.to(dpct_device), clip_encoded_txt.to(dpct_device)

            # concatenating clip_encoded_txt and clip_txt_emb for interfacing with other functions
            clip_txt_emb = clip_txt_emb.unsqueeze(1) # clip_txt_emb.shape: [batch_size, 1, d_model]
            clip_txt_out = torch.cat((clip_encoded_txt, clip_txt_emb), dim=1) # clip_txt_out.shape: [batch_size, max_seq_len_clip + 1, d_model]

            # get sampled_clip_img_emb from the diffusion_prior
            sampled_clip_img_emb = sample_strided_CFG(dpct_model, alphas_hat, guidance_strength, max_time_steps, subseq_steps, clip_img_emb.shape[1:], device, batch_size, clip_txt_out, cfg_flag=False, pred_x_start=True, progress_bar=False) # sampled_clip_img_emb.shape: [batch_size, d_model]

            # move from dpct_device to (decoder) device
            sampled_clip_img_emb = sampled_clip_img_emb.to(device)

            # set sampled_clip_img_emb = None with prob p_uncond
            sampled_clip_img_emb_label = sampled_clip_img_emb[0].unsqueeze(0) # pick one item from the batch - to be used while sampling
            if np.random.rand() < p_uncond:
                sampled_clip_img_emb = None

            t = torch.randint(low=1, high=max_time_steps, size=(batch_size,)).to(device) # sample a time step uniformly in [1, max_time_steps)

        loss = calculate_hybrid_loss(decoder, imgs_unet, t, sampled_clip_img_emb, L_lambda, alphas_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % (max_epochs//200) == 0:
            print('ep:{} \t loss:{}'.format(ep, loss.item()))

            # save checkpoint
            save_ckpt(ckpt_path_glide, decoder, optimizer)

            # sample
            decoder.eval()

            sample_caption_text = minibatch[0][1]
            sampled_img = sample_strided_CFG(decoder, alphas_hat, guidance_strength, max_time_steps, subseq_steps, imgs_unet.shape[1:], device, 1, sampled_clip_img_emb_label, cfg_flag=True)
            # save sampled_img
            save_img_CFG(sampled_img, str(ep) + ': ' + sample_caption_text + '.png')

            decoder.train()
