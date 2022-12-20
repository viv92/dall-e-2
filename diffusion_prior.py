### Program implementing the diffusion based prior for unclip

## Features:
# 1. The diffusion process is similar to that in the "improved_ddpm" paper (learnt mean and covariance, cfg)
# 2. The prior uses a causal transformer_encoder to parameterize the diffusion model (instead of a UNet)
# 3. The input embedding sequence to the causal transformer is [clip_encoded_txt, clip_txt_emb, diffusion_timestep_emb, noised_clip_img_emb, final_emb]
# 4. From the transformer output sequence, the embeddings corresponding to the final_emb are used to predict the unnoised_clip_img_emb (thus forming the diffusion loss)

# 5. [IMPORTANT DIFFERENCE FROM TRADITIONAL DDPM] The diffusion loss in the dalle2's diffusion prior is MSE loss between the predicted unnoised clip_img_emb from the causal transformer and the true unnoised clip_img_emb. Note that this different from the loss used in the traditional DDPM paper, in which, we can either (1) take a mse over predicted mean u_theta and surrogate posterior mean u_tilde; or (2) take mse over predicted noise and true noise. ( The loss in dalle2's diffusion prior corresponds to mse over x_start and predicted x_start. The predicted x_start can be calculated from predicted noise and input x_t using the formula from the forward process - though this formula is valid for the reverse process only at convergence, this way we are encouraging the reverse process to match the forward process - similar in spirit to mse over predicted noise and true noise or mse over predicted mean and surrogate mean)

## Todos / Questions:
# 1. ensure that the predicted unnoised clip_img_emb from the causal transformer are in range [-1, 1] - required by the diffusion process
# 2. In DPCT_Embeddings, try fixed sinusoidal embeddings for time step
# 3. p_uncond is currently set to zero (no classifier-free guidance) as masking the conditioning_embedding results in a padding mask that is left aligned (masking all columns except the last 3). This leads to NaN values in attention weight matrix. - Not sure how to do drop the conditioning_embedding without encountering NaNs.


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sentencepiece as spm

from utils_transformer import *
from utils_diffusion import *
from CLIP_resnet import ImageEmbeddings, CaptionEmbeddings, ClipTextEncoder, CLIP, init_text_encoder, init_CLIP, load_data, forward_hook, process_batch

# class for converting inputs to DPCT (diffusion prior causal transformer) as valid input embeddings sequence
# this involves doing the following:
# 1. Recall that the input to DPCT is a sequence [clip_encoded_txt, clip_txt_emb, diffusion_timestep_emb, noised_clip_img_emb, final_emb]
# 2. clip_encoded_txt and clip_txt_emb are already in embedding form (nothing to be done for these )
# 3. time step is just an int. So we learn an embedding for it in the form of a nn.Linear layer (todo: try fixed sinusoidal embeddings for time step)
# 4. final_emb is just a token (like <cls>). We learn a nn.Parameter for it as the embedding
# 5. add positional encoding to the entire input sequence
class DPCT_Embeddings(nn.Module):
    def __init__(self, max_seq_len_dpct, d_model, dropout, device):
        super().__init__()
        self.final_emb = nn.Parameter(torch.ones(d_model)) # learnable embedding for dummy <final> token
        self.pos_emb = nn.Embedding(max_seq_len_dpct, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_len_dpct = max_seq_len_dpct
        self.device = device
    # function to get time embeddings from time int (based on sinusoidal position encoding)
    def get_time_embedding(self, t, d_model): # t.shape: [batch_size]
        t = t.unsqueeze(-1).float()
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, d_model, 2, device=self.device).float() / d_model)
        )
        pos_enc_a = torch.sin(t.repeat(1, d_model // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, d_model // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    # function for forward prop
    def forward(self,
                clip_img_emb,    # clip_img_emb.shape: [batch_size, d_model]
                t,              # t.shape: [batch_size]
                encoded_txt,    # encoded_txt.shape: [batch_size, seq_len, d_model]
                clip_txt_emb):   # clip_txt_emb.shape: [batch_size, d_model]
        batch_size = encoded_txt.shape[0]
        t_emb = self.get_time_embedding(t, self.d_model) # t_emb.shape: [batch_size, d_model]
        final_emb = self.final_emb.unsqueeze(0).expand(batch_size, -1) # final_emb.shape: [batch_size, d_model]
        positions = torch.arange(self.max_seq_len_dpct).to(self.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1) # positions.shape: [batch_size, max_seq_len_dpct]
        pos_emb = self.pos_emb(positions) # pos_emb.shape: [batch_size, max_seq_len_dpct, d_model]
        clip_txt_emb = clip_txt_emb.unsqueeze(1) # clip_txt_emb.shape: [batch_size, 1, d_model]
        clip_img_emb = clip_img_emb.unsqueeze(1) # clip_img_emb.shape: [batch_size, 1, d_model]
        t_emb = t_emb.unsqueeze(1) # t_emb.shape: [batch_size, 1, d_model]
        final_emb = final_emb.unsqueeze(1) # final_emb.shape: [batch_size, 1, d_model]
        dpct_input_emb = torch.cat((encoded_txt, clip_txt_emb, t_emb, clip_img_emb, final_emb), dim=1) # dpct_input_emb.shape: [batch_size, max_seq_len_dpct, d_model]
        dpct_input_emb = self.dropout( self.norm(dpct_input_emb + pos_emb) )
        return dpct_input_emb

# class implementing the diffusion prior causal transformer (DPCT) - used as the denoising network for the diffusion prior
class DPCT(nn.Module):
    def __init__(self, embedder, encoder, max_seq_len_dpct, d_model, device):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.max_seq_len = max_seq_len_dpct
        self.d_model = d_model
        self.mask_causal = self.make_causal_mask(max_seq_len_dpct).to(device)
        self.mask_padding = self.make_pad_mask(max_seq_len_dpct).to(device) # used to mask conditioning signal with a prob p_uncond (for classifier-free guidance)
        self.final_proj = nn.Linear(d_model, d_model*2) # doubling the channels for both learnable mean and variance of the diffusion distribution
        self.device = device
    def make_causal_mask(self, max_seq_len):
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8)
        return mask == 1  # True elements are masked
    def make_pad_mask(self, max_seq_len):
        mask = torch.ones(max_seq_len).type(torch.uint8)
        mask[-3:] = 0 # unmask t_emb, clip_img_emb, final_emb
        return mask == 1
    def forward(self, clip_img_emb, t, clip_txt_out):
        batch_size = clip_img_emb.shape[0]
        if clip_txt_out is None: # set to None with probability p_uncond (for classifier-free guidance)
            # init dummy encoded_txt and clip_txt_emb and set pad mask for masking the dummy input indices
            encoded_txt = torch.zeros(batch_size, self.max_seq_len-4, self.d_model).to(self.device)
            clip_txt_emb = torch.zeros(batch_size, self.d_model).to(self.device)
            mask_padding = self.mask_padding # mask_padding.shape: [max_seq_len]
            mask_padding = mask_padding.expand(batch_size, -1) # mask_padding.shape: [batch_size, max_seq_len]
        else:
            encoded_txt, clip_txt_emb = clip_txt_out[:, :-1], clip_txt_out[:, -1]
            mask_padding = None
        dpct_input_emb = self.embedder(clip_img_emb, t, encoded_txt, clip_txt_emb)
        dpct_out_seq = self.encoder(dpct_input_emb, mask_padding=mask_padding, mask_causal=self.mask_causal) # dpct_out_seq.shape: [batch_size, max_seq_len_dpct, d_model]
        dpct_final_out = dpct_out_seq[:, -1] # dpct_final_out.shape: [batch_size, d_model]
        dpct_final_out = self.final_proj(dpct_final_out) # dpct_final_out.shape: [batch_size, d_model * 2]
        return dpct_final_out # the predicted denoised clip_img_emb

# caller function to init dpct
def init_dpct(max_seq_len_dpct, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    dpct_embeddings = DPCT_Embeddings(max_seq_len_dpct, d_model, dropout, device)
    model = DPCT(dpct_embeddings, encoder, max_seq_len_dpct, d_model, device)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() # clip is only used for inference
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer

# # utility function to load checkpoint to resume training of dpct - loads to the device passed as 'device' argument
# def load_ckpt_dpct(checkpoint_path, model, optimizer=None, device=torch.device('cpu'), mode='eval'):
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(ckpt['model_state_dict'])
#     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#     if mode == 'eval':
#         model.eval()
#         return model
#     else:
#         model.train() # put dpct to train mode
#         return model, optimizer

# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on whatever device the model was training on
def save_ckpt(checkpoint_path, model, optimizer, scheduler=None):
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)



### main - to setup and train DPCT
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
    p_uncond = 0. # probability for setting conditioning_embedding = None
    beta_min = 1e-4 # not needed for cosine noise schedule
    beta_max = 0.02 # not needed for cosine noise schedule
    max_time_steps = 1000 # 4000
    subseq_steps = 200 # used for both strided sampling and ddim sampling - should be less than max_time_steps

    img_size = 224 # resize for resnet
    lr = 3e-4
    batch_size = 256
    max_epochs = 30000 * 1
    random_seed = 10

    ckpt_path_clip = 'ckpts_clip_resnet/latest.pt'

    ckpt_path_dpct = 'ckpts_dpct/latest.pt'
    resume_training_from_ckpt = True

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # clip_device = torch.device('cpu') # clip model will be loaded for inference on the cpu (to save memory on the gpu for diffusion prior model)
    clip_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load spm as tokenizer
    spm_processor = spm.SentencePieceProcessor(model_file='spm1.model')
    vocab_size_clip = len(spm_processor)

    # declare sos, eos, pad tokens
    sos_token, eos_token, unk_token = spm_processor.piece_to_id(['<s>', '</s>', '<unk>'])
    pad_token = unk_token # <unk> token is used as the <pad> token

    # load data and create img_cap_dict
    img_cap_dict, max_seq_len_clip = load_data()

    # init clip model
    clip_image_encoder = ImageEmbeddings(forward_hook, d_model).to(clip_device) # resnet
    clip_text_encoder = init_text_encoder(vocab_size_clip, max_seq_len_clip, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, clip_device).to(clip_device)
    clip_model = init_CLIP(clip_text_encoder, clip_image_encoder, d_model, eos_token).to(clip_device)
    # load clip_model weights and put clip_model in eval mode
    clip_model = load_ckpt(ckpt_path_clip, clip_model, device=clip_device, mode='eval')

    # init DPCT
    max_seq_len_dpct = max_seq_len_clip + 4 # for clip_txt_emb, time_step, clip_img_emb, final_emb
    net = init_dpct(max_seq_len_dpct, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # calcualate betas and alphas
    betas = linear_noise_schedule(beta_min, beta_max, max_time_steps)
    # betas = cosine_noise_schedule(max_time_steps)
    # betas = betas_for_alpha_bar(max_time_steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alphas_hat = alphas_hat
    alphas_hat = alphas_hat.to(device) # alphas_hat.shape: [max_time_steps]

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr)

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        net, optimizer = load_ckpt(ckpt_path_dpct, net, optimizer, device=device, mode='train')

    # train
    for ep in tqdm(range(max_epochs)):

        # fetch minibatch - a batch of [img_filename, caption_text] pairs
        minibatch_keys = np.random.choice(list(img_cap_dict.keys()), size=batch_size)
        minibatch = [[k, img_cap_dict[k]] for k in minibatch_keys]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        # note that we don't create embeddings yet, since that's done by the image_encoder and text_encoder in the CLIP model
        imgs, cap_tokens, cap_mask_padding, cap_mask_causal = process_batch(minibatch, clip_model, spm_processor, sos_token, eos_token, pad_token, max_seq_len_clip, img_size, clip_device) # imgs.shape:[batch_size, 3, 64, 64], captions.shape:[batch_size, max_seq_len_clip]

        with torch.no_grad():
            # encode (inference) through CLIP
            # clip_img_emb.shape: [batch_size, d_model]
            # clip_txt_emb.shape: [batch_size, d_model]
            # clip_encoded_txt.shape: [batch_size, max_seq_len_clip, d_model]
            clip_img_emb, clip_txt_emb, clip_encoded_txt = clip_model.encode(imgs, cap_tokens, cap_mask_padding, cap_mask_causal)

            # move from cpu (clip_device) to gpu
            clip_img_emb, clip_txt_emb, clip_encoded_txt = clip_img_emb.to(device), clip_txt_emb.to(device), clip_encoded_txt.to(device)

            # concatenating clip_encoded_txt and clip_txt_emb for interfacing with other functions
            clip_txt_emb = clip_txt_emb.unsqueeze(1) # clip_txt_emb.shape: [batch_size, 1, d_model]
            clip_txt_out = torch.cat((clip_encoded_txt, clip_txt_emb), dim=1) # clip_txt_out.shape: [batch_size, max_seq_len_clip + 1, d_model]

            # set clip_txt_out = None with prob p_uncond
            clip_txt_label = clip_txt_out[0].unsqueeze(0) # pick one item from the batch - to be used while sampling
            if np.random.rand() < p_uncond:
                clip_txt_out = None

            t = torch.randint(low=1, high=max_time_steps, size=(batch_size,)).to(device) # sample a time step uniformly in [1, max_time_steps)

        loss = calculate_hybrid_loss(net, clip_img_emb, t, clip_txt_out, L_lambda, alphas_hat, mse_over_x_start=True, pred_x_start=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 150 == 0:
            print('ep:{} \t loss:{}'.format(ep, loss.item()))

            # save checkpoint
            save_ckpt(ckpt_path_dpct, net, optimizer)

            # # sample
            # net.eval()
            #
            # sample_caption_text = minibatch[0][1]
            # sampled_clip_img_emb = sample_strided_CFG(net, alphas_hat, guidance_strength, max_time_steps, subseq_steps, clip_img_emb.shape[1:], device, 1, clip_txt_label, cfg_flag=False, pred_x_start=True)
            # # verify
            # print('--------- sampled_clip_img_emb.shape: ', sampled_clip_img_emb.shape)
            #
            # net.train()
