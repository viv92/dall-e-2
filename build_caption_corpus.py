### Utility to extract all captions from the dataset and build a caption corpus txt file

import os
import json
import unidecode


imgs_folder = 'dataset_coco_val2017/images/'
captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
captions_file = open(captions_file_path)
captions = json.load(captions_file)

img_dict, caption_dict, img_cap_dict = {}, {}, {}
all_captions = []

for img in captions['images']:
    id, file_name = img['id'], img['file_name']
    img_dict[id] = file_name

for cap in captions['annotations']:
    id, caption = cap['image_id'], cap['caption']

    # process caption
    caption = unidecode.unidecode(caption) # strip accents
    caption = caption.lower() # lowercase

    caption_dict[id] = caption
    img = img_dict[id]
    img_cap_dict[img] = caption
    all_captions.append(caption)

## build caption corpus file to train sentencepiece model
corpus = '\n'.join(all_captions)

with open('caption_corpus_val2017.txt', 'w+') as caption_corpus_file:
    caption_corpus_file.write(corpus)
