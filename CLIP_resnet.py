### Program implementing CLIP (Contrastive Language Image Pretraining) - using ResNet as image encoder

## Features:
# 1. Multi-class N pair loss / contrastive loss over minibatch of image-text embedding pairs.

## Todos / Questions:
# 1. Image encoder: Vision Transformer versus ResNet
# 2. Text encoder: Transformer encoder with subsequent mask (masked self-attention for promoting language modelling skills in the encoder)
# 2.1 So do we have an auxiliary loss for language modelling along with the clip loss ? (paper does not use an explicit auxiliary loss, but the masked self-attention is supposed to implicitly promote language modelling abilities in transformer encoder)
# 2.2 Swapping the order of layernorm in sublayer connection to apply layernorm first (this would require a final layer norm in the encoder out)
# 2.3. Text encoding = encoder output for the [EOS] index (not using the entire ouput vector of shape [max_seq_len, vocab_size])
# 2.4. Another version - use BERT encoder: Bert style masking instead of subsequent mask in transformer encoder
# 3. add layernorm and dropout to image embeddings (as done for text embeddings)
# 4. logits are pairwise cosine similarities, scalled by exp(temperature). Temperature is a learnt parameter, not a tuned hyperparameter.
# 5. Contrastive loss versus CrossEntropyLoss - similarities and differences: refer register note.
# 6. GELU non-linearity in the feedforward layers of transformer
# 7. Replace avg pooling in resnet with attention pooling
# 8. Optimize code - faster vector normalization
# 9. cross check with moen's code.
# 10. img preprocessing: does it make a difference if img.shape: [c,h,w] versus [c,w,h] - What does resnet / ViT expect ?
# 11. Why test_trials > 1 give CUDA_OUT_OF_MEMORY error ?
# 12. Save state of lr_scheduler when saving checkpoint


import os
import math
import numpy as np
import cv2
import json
import unidecode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import sentencepiece as spm

# for learning rate scheduling
from torch.optim.lr_scheduler import LambdaLR

# import transformer modules
from utils_transformer import *

# forward hook for reading resnet penultimate layer logits
def forward_hook(module, input, output):
    global resnet_avgpool_output
    resnet_avgpool_output = output

# class for image embeddings
class ImageEmbeddings(nn.Module):
    def __init__(self, hook_fn, d_model):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.resnet.avgpool.register_forward_hook(hook_fn)
        self.proj = nn.Linear(2048, d_model, bias=False)
    def forward(self, imgs): # imgs.shape: [b,c,w,h]
        _ = self.resnet(imgs)
        emb = resnet_avgpool_output # emb.shape: [b, 2048, 1, 1]
        emb = emb.flatten(start_dim=1, end_dim=-1) # emb.shape: [b, 2048]
        emb = self.proj(emb) # emb.shape: [b, d_model]
        return emb

# class for caption embeddings
class CaptionEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, dropout, device):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.device = device
    def forward(self, x):
        batch_size, max_seq_len = x.shape[0], x.shape[1]
        tok_emb = self.tok_emb(x) # tok_emb.shape: [batch_size, max_seq_len, d_model]
        positions = torch.arange(max_seq_len).to(self.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        final_emb = self.dropout( self.norm(tok_emb + pos_emb) )
        final_emb = final_emb * math.sqrt(self.d_model)
        return final_emb

# class implementing the text_encoder
class ClipTextEncoder(nn.Module):
    def __init__(self, encoder, caption_embeddings):
        super().__init__()
        self.caption_embeddings = caption_embeddings
        self.encoder = encoder
    def forward(self, cap_tokens, cap_mask_padding, cap_mask_causal):
        cap_embs = self.caption_embeddings(cap_tokens)
        encoded_text = self.encoder(cap_embs, cap_mask_padding, cap_mask_causal)
        return encoded_text

# class implemeting the entire model
class CLIP(nn.Module):
    def __init__(self, text_encoder, image_encoder, d_model, eos_token):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.eos_proj = nn.Linear(d_model, d_model, bias=False) # this is for encoded_txt (we don't need this for encoded_img as resnet handles that)
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.eos_token = eos_token

    def encode_text(self, cap_tokens, cap_mask_padding, cap_mask_causal):
        encoded_txt = self.text_encoder(cap_tokens, cap_mask_padding, cap_mask_causal) # encoded_txt.shape: [batch_size, max_seq_len, d_model]
        # get eos indices in cap_tokens
        eos_idx = torch.where(cap_tokens == self.eos_token) # eos_idx is tupe of shape (batch_size, 2)
        # extract eos index as representation for encoded_txt
        encoded_txt_eos = encoded_txt[eos_idx]
        # project encoded_img to common embedding space (note that this is not needed for encoded_img as resnet takes care of this)
        clip_txt_emb = self.eos_proj(encoded_txt_eos) # encoded_txt.shape: [batch_size, d_model]
        return clip_txt_emb, encoded_txt

    def encode(self, imgs, cap_tokens, cap_mask_padding, cap_mask_causal):
        clip_img_emb = self.image_encoder(imgs)
        clip_txt_emb, encoded_txt = self.encode_text(cap_tokens, cap_mask_padding, cap_mask_causal)
        return clip_img_emb, clip_txt_emb, encoded_txt

    def forward(self, imgs, cap_tokens, cap_mask_padding, cap_mask_causal):
        clip_img_emb, clip_txt_emb, encoded_txt = self.encode(imgs, cap_tokens, cap_mask_padding, cap_mask_causal) # clip_img_emb.shape: [batch_size, d_model]; encoded_txt.shape: [batch_size, seq_len, d_model]
        # cosine similarity score matrix
        clip_img_emb_norm = torch.linalg.vector_norm(clip_img_emb, dim=1).unsqueeze(1) # shape: [batch_size, 1]
        clip_txt_emb_norm = torch.linalg.vector_norm(clip_txt_emb, dim=1).unsqueeze(1)
        norm_matrix = torch.matmul(clip_img_emb_norm, clip_txt_emb_norm.T)
        cosine_similarity = torch.matmul(clip_img_emb, clip_txt_emb.T) / norm_matrix
        scores = cosine_similarity * torch.exp(self.temperature)
        return scores

    # function used for test time label prediction (1 = correct pair, 0 = wrong pair)
    def predict(self, imgs, cap_tokens, cap_mask_padding, cap_mask_causal, threshold=0.9):
        scores = self.forward(imgs, cap_tokens, cap_mask_padding, cap_mask_causal)
        scores = scores.trace()
        preds = torch.gt(scores, threshold).int()
        return preds


# caller function to init the text encoder (transformer_encoder)
def init_text_encoder(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    caption_embeddings = CaptionEmbeddings(vocab_size, max_seq_len, d_model, dropout, device)
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    model = ClipTextEncoder(encoder, caption_embeddings)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# caller function to instantiate the CLIP model, using the defined hyperparams as input
def init_CLIP(text_encoder, image_encoder, d_model, eos_token):
    model = CLIP(text_encoder, image_encoder, d_model, eos_token) # the CLIP model
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# utility function to load img and captions data
def load_data():
    imgs_folder = 'dataset_coco_val2017/images/'
    captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)

    img_dict, img_cap_dict = {}, {}
    max_caption_len = 0

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name

    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # update max_caption_len
        caption_len = len(caption.split(' '))
        if caption_len > max_caption_len:
            max_caption_len = caption_len
        # process caption
        caption = unidecode.unidecode(caption) # strip accents
        caption = caption.lower()
        # use img_name as key for img_cap_dict
        img = img_dict[id]
        img_cap_dict[img] = caption

    max_caption_len += 3 # for <s>, </s> and a precautionary <pad>
    return img_cap_dict, max_caption_len


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - then obtain embeddings for them
def process_batch(minibatch, model, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device):
    augmented_imgs, tokenized_captions = [], []
    imgs_folder = 'dataset_coco_val2017/images/'
    for img_filename, caption_text in minibatch:
        # tokenize caption text
        caption_tokens = spm_processor.encode(caption_text, out_type=int)
        caption_tokens = [sos_token] + caption_tokens + [eos_token] # append sos and eos tokens
        while len(caption_tokens) < max_seq_len:
            caption_tokens.append(pad_token) # padding
        tokenized_captions.append(caption_tokens)
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
    tokenized_captions = torch.LongTensor(tokenized_captions).to(device)
    # get caption mask
    cap_mask_padding = pad_mask(tokenized_captions, pad_token) # pad mask for captions
    cap_mask_causal = subsequent_mask(tokenized_captions.shape).to(device)
    return augmented_imgs, tokenized_captions, cap_mask_padding, cap_mask_causal

# function to calculate loss by forward propping through the model
def calculate_loss(model, imgs, cap_tokens, cap_mask_padding, cap_mask_causal, batch_size, device):
    # feed embeddings to transformer decoder
    scores = model(imgs, cap_tokens, cap_mask_padding, cap_mask_causal) # scores.shape: [batch_size, batch_size]
    # targets
    targets = torch.arange(batch_size).long().to(device)
    # cross entropy loss
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_img = criterion(scores, targets)
    loss_txt = criterion(scores.T, targets)
    loss = (loss_img + loss_txt) / 2
    return loss

# utility function for learning rate schedule - from attention paper
def lr_rate(curr_step, warmup_steps, lr_init, d_model):
    curr_step += 1 # one-indexed instead of zero indexed
    lr_new = lr_init * ( d_model**(-0.5) * min( curr_step**(-0.5), curr_step * (warmup_steps**(-1.5)) ) )
    return lr_new

# function to calculate test accuracy
def calculate_test_accuracy(test_trials, model, test_data):
    test_accuracy = []
    for ep in range(test_trials):
        # fetch minibatch
        idx = np.arange(len(test_data))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [test_data[i] for i in idx]
        gt_labels = np.ones(batch_size)
        # flip labels at random
        for i in range(batch_size):
            r = np.random.choice(np.arange(2))
            if r == 0:
                # flip label
                gt_labels[i] = 0
                # flip caption
                flipped_idx = int((i+1) % batch_size)
                minibatch[i][1] = minibatch[flipped_idx][1]
        # get predicted labels
        threshold = 0.9
        # process batch to get batch of imgs, tokenized_captions and masks
        imgs, cap_tokens, cap_mask_padding, cap_mask_causal = process_batch(minibatch, model, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device)
        # feed to transformer decoder
        pred_labels = model.predict(imgs, cap_tokens, cap_mask_padding, cap_mask_causal, threshold) # pred_labels.shape: [batch_size]
        # accuracy
        gt_labels = torch.from_numpy(gt_labels).int().to(device)
        accuracy = torch.eq(gt_labels, pred_labels.detach()).int().sum()
        accuracy = (accuracy.item() * 100) / batch_size
        test_accuracy.append(accuracy)
    mean_accuracy = sum(test_accuracy) / len(test_accuracy)
    return mean_accuracy

# utility function to save a checkpoint (model_state, optimizer_state, scheduler state) - saves on whatever device the model was training on
def save_checkpoint_clip(checkpoint_path, model, optimizer, lr_scheduler):
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()}
    torch.save(save_dict, checkpoint_path)

# utility function to load checkpoint to resume training from - loads to the device passed as 'device' argument
def load_checkpoint_clip(checkpoint_path, model, optimizer, lr_scheduler, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    model.train()
    return model, optimizer, lr_scheduler


### main ###
if __name__ == '__main__':

    # hperparams
    img_size = 224 # resize for resnet
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 16
    train_split = .95
    test_trials = 10
    num_epochs = 24000 * 12
    random_seed = 10
    lr_init = 1 # using lr_scheduler
    warmup_steps = 4000

    checkpoint_path = 'ckpts_clip_resnet/latest.pt' # path to a saved checkpoint (model state, optimizer state, scheduler state) to resume training from
    resume_training_from_ckpt = True

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # spm processor as tokenizer / detokenizer
    spm_processor = spm.SentencePieceProcessor(model_file='spm1.model')
    vocab_size = len(spm_processor)

    # load data and create img_cap_dict
    img_cap_dict, max_seq_len = load_data()

    # tokenize captions and append sos, eos, pad tokens
    sos_token, eos_token, unk_token = spm_processor.piece_to_id(['<s>', '</s>', '<unk>'])
    pad_token = unk_token # <unk> token is used as the <pad> token

    # train-test split
    train_data, test_data = [], []
    n_total_keys = len(img_cap_dict.keys())
    n_train_keys = int(n_total_keys * train_split)
    for i, (k, v) in enumerate(img_cap_dict.items()):
        if i <= n_train_keys:
            train_data.append([k, v])
        else:
            test_data.append([k, v])

    # free memory occupied by img_cap_dict as its no longer needed
    del img_cap_dict

    # init image encoder model (resnet)
    image_encoder = ImageEmbeddings(forward_hook, d_model).to(device)

    # init text_encoder model (transformer encoder)
    text_encoder = init_text_encoder(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # init CLIP model
    model = init_CLIP(text_encoder, image_encoder, d_model, eos_token).to(device)

    # instantiate optimizer and lr_scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_init, betas=(.9, .98), eps=1e-9, weight_decay=1e-4)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: lr_rate(x, warmup_steps, lr_init, d_model) ) # x increases like a counter on each call to lr_scheduler.step()

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        model, optimizer, lr_scheduler = load_checkpoint_clip(checkpoint_path, model, optimizer, lr_scheduler, device)

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(n_train_keys)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [train_data[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        # note that we don't create embeddings yet, since that's done by the image_encoder and text_encoder in the CLIP model
        imgs, cap_tokens, cap_mask_padding, cap_mask_causal = process_batch(minibatch, model, spm_processor, sos_token, eos_token, pad_token, max_seq_len, img_size, device) # imgs.shape:[batch_size, 3, 64, 64], captions.shape:[batch_size, max_seq_len]

        # calculate loss
        loss = calculate_loss(model, imgs, cap_tokens, cap_mask_padding, cap_mask_causal, batch_size, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


        if ep % 1000 == 0:

            # print intermediate loss
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

            # save checkpoint
            save_checkpoint_clip(checkpoint_path, model, optimizer, lr_scheduler)

            # test - get test accuracy
            model.eval()
            accuracy = calculate_test_accuracy(test_trials, model, test_data)
            print('accuracy: ', accuracy)
            model.train()
