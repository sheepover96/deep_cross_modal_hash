import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import time
from torchtext.data import Field
from torchtext.datasets import LanguageModelingDataset
import spacy
spacy_en = spacy.load('en')
from nltk.corpus import stopwords
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from data_loader import DcmhDataset
from dcmh_model import CNNModel, TextModel




device = torch.device('cpu')

HASH_CODR_LENGTH = 32
IMG_SIZE = 256
BATCH_SIZE = 256
NEPOCH = 1000

def calc_map(queries, queries_label, target_item, target_label, k=None):
    mAP = 0.
    if k is None:
        k = target_item.shape[0]
    for query_item, query_label in zip(queries, queries_label):

        is_correct_target = (torch.matmul(query_label, target_label.t()) > 0).type(torch.FloatTensor)
        correct_target_num = is_correct_target.sum().type(torch.LongTensor).item()
        if correct_target_num == 0:
            continue

        hamming_dist = torch.matmul(query_item, target_item.t())
        _, hd_sorted_idx = hamming_dist.sort(descending=True)
        query_result = is_correct_target[hd_sorted_idx]
        total = min(k, correct_target_num)

        count = torch.arange(1, correct_target_num+1)
        tindex = torch.nonzero(query_result)[:total].squeeze() + 1.
        mAP += torch.mean(count.type(torch.FloatTensor)/tindex.type(torch.FloatTensor))
    return mAP/len(queries)

def tokenizer2(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_query_result(query, target_item):
    hamming_dist = torch.matmul(query, target_item.t())
    _, hd_sorted_idx = hamming_dist.sort(descending=True)
    return hd_sorted_idx

def generate_img_code(img_model, query):
    model_out =  img_model(query)
    hash_code = torch.sign(model_out)
    return hash_code

def generate_text_code(text_model, query):
    model_out =  text_model(query)
    hash_code = torch.sign(model_out)
    return hash_code

en_stopwords = stopwords.words('english')
TAG_TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True, stop_words=['<eos>'])
tag_lang = LanguageModelingDataset(path='./iapr_tags.txt', text_field=TAG_TEXT)
TAG_TEXT.build_vocab(tag_lang)
tag_vocab = TAG_TEXT.vocab
tag_vocab_size = len(tag_vocab.stoi) + 1
#print(vocab.itos)
#print(vocab.stoi)
DESC_TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True, stop_words=(['<eos>'] + en_stopwords))
desc_lang = LanguageModelingDataset(path='./iapr_docs.txt', text_field=DESC_TEXT)
DESC_TEXT.build_vocab(desc_lang, min_freq=2)
desc_vocab = DESC_TEXT.vocab
desc_vocab_size = len(desc_vocab.stoi) + 1

train_data = DcmhDataset('./train_saiapr2.csv', tag_vocab.stoi, tag_vocab_size, desc_vocab.stoi, desc_vocab_size)
test_data = DcmhDataset('./test_saiapr2.csv', tag_vocab.stoi, tag_vocab_size, desc_vocab.stoi, desc_vocab_size)

img_model = CNNModel(IMG_SIZE, HASH_CODR_LENGTH)
img_model.load_state_dict(torch.load('model/img_model_desc.t7', map_location=device))
text_model = TextModel(desc_vocab_size, HASH_CODR_LENGTH)
text_model.load_state_dict(torch.load('model/text_model_desc.t7', map_location=device))

print('loaded')


ntest = len(test_data)

idx_list = [i for i in range(ntest)]
query_idx_list = np.random.choice(idx_list, 20, replace=False)
ret_idx_list = idx_list
query_data = [ test_data[idx] for idx in query_idx_list ] 
ret_data = [ test_data[idx] for idx in ret_idx_list ] 

query_loader = torch.utils.data.DataLoader(query_data, batch_size=len(query_data))
for data_ids, imgs, doc_vecs, tag_vecs, desc in query_loader:
    imgs, doc_vecs, tag_vecs = imgs.to(device), doc_vecs.to(device), tag_vecs.to(device)
    query_img_hash_code = generate_img_code(img_model, imgs)
    query_text_hash_code = generate_text_code(text_model, doc_vecs)
    query_labels = tag_vecs

ret_loader = torch.utils.data.DataLoader(ret_data, batch_size=len(ret_data))
for data_ids, imgs, doc_vecs, tag_vecs, desc in ret_loader:
    imgs, doc_vecs, tag_vecs = imgs.to(device), doc_vecs.to(device), tag_vecs.to(device)
    print(doc_vecs.shape)
    ret_img_hash_code = generate_img_code(img_model, imgs)
    ret_text_hash_code = generate_text_code(text_model, doc_vecs)
    ret_labels = tag_vecs

# img to text
#mAP = calc_map(query_img_hash_code, query_labels, ret_text_hash_code, ret_labels)
#print(mAP)
for idx, (img_hash, text_hash, (data_ids, imgs, doc_vecs, tag_vecs, desc)) in\
        enumerate(zip(query_img_hash_code, query_text_hash_code, query_loader)):
    data_id, img, doc_vec, tag_vec, desc = query_data[idx]
    print('answer: ', desc)
    print(ret_text_hash_code.shape)
    print(img_hash.shape)
    sorted_idx = get_query_result(img_hash, ret_text_hash_code)
    for desc_idx in sorted_idx:
        print(ret_data[desc_idx][4])
    plt.imshow(img.transpose(0,2).numpy())
    plt.show()


# text to image
for img_hash, text_hash, (data_ids, imgs, doc_vecs, tag_vecs, desc) in\
        zip(query_img_hash_code, query_text_hash_code, query_loader):
    sorted_idx = get_query_result(img_hash, ret_text_hash_code)
    no1_text = ret_data[sorted_idx[0]][2]
    no2_text = ret_data[sorted_idx[1]][2]
    
print('finish')
