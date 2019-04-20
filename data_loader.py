import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import spacy
spacy_en = spacy.load('en')
import pandas as pd
import ast
from PIL import Image
 
class ResizeImg():

    def __init__(self, resized_len=256.):
        self.resized_len = resized_len 

    def __call__(self, img):
        w, h = img.size
        if h <= w:
            short_side_len = h
            resize_ratio = self.resized_len/short_side_len
            resized_w = int(w * resize_ratio)
            resized_h = 256
        else:
            short_side_len = w
            resize_ratio = self.resized_len/short_side_len
            resized_w = 256
            resized_h = int(h * resize_ratio)
        
        resized_img = img.resize((resized_w, resized_h))
        return resized_img


class DcmhDataset(Dataset):

    def __init__(self, data_path, vocab_stoi, vocab_size, desc_stoi, desc_size, *args, **kwargs):
        self.df = pd.read_csv(data_path, header=None)
        self.vocab_size = vocab_size
        self.vocab_stoi = vocab_stoi
        self.desc_stoi = desc_stoi
        self.desc_size = desc_size
        self.img_transform = transforms.Compose(
            [ResizeImg(), transforms.RandomCrop(256), transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def _tokenizer(self, text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    def __getitem__(self, idx):
        data_id = self.df.iat[idx, 0]
        img_path = self.df.iat[idx, 1]
        img = Image.open(img_path)
        if self.img_transform:
            img = self.img_transform(img)#.view(256,256,3)

        desc_vec = torch.zeros(self.desc_size)
        description = self.df.iat[idx, 2]
        desc_token = self._tokenizer(description)
        for letter in description:
            if letter in self.desc_stoi:
                desc_vec[self.desc_stoi[letter]] = 1.

        tag_list = ast.literal_eval(self.df.iat[idx, 3])
        tag_vec = torch.zeros(self.vocab_size)
        for tag in tag_list:
            tag_vec[self.vocab_stoi[tag]] = 1.
        if tag_vec.sum() == 0:
            print(tag_vec)

        return [data_id, img, desc_vec, tag_vec, description]
