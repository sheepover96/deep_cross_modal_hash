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
import nltk
from nltk.corpus import stopwords
import spacy
spacy_en = spacy.load('en')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from data_loader import DcmhDataset
from dcmh_model import CNNModel, TextModel


GPU = torch.cuda.is_available()

HASH_CODR_LENGTH = 32
IMG_SIZE = 256
BATCH_SIZE = 256
NEPOCH = 1000


def train_cnn(epoch, img_model, text_model, hash_matrix, sim_matrix, device, train_loader, img_optimizer, img_out, text_out, gamma, eta, ntrains):
    #img_model.train()
    #text_model.eval()

    loss_mean = 0
    for batch_idx, (data_ids, data_idxs, imgs, desc_vecs, tag_vecs) in enumerate(train_loader):
        batch_size = len(data_ids)
        imgs, desc_vecs, tag_vecs = imgs.to(device), desc_vecs.to(device), tag_vecs.to(device)
        unupdated_ids = np.setdiff1d(range(ntrains), data_ids)
        ones = torch.ones(batch_size, 1).to(device)
        ones_ = torch.ones(ntrains - batch_size, 1).to(device)

        img_optimizer.zero_grad()
        img_out_batch = img_model(imgs)
        #print(img_out_batch)

        img_out[data_ids,:] = img_out_batch.data

        theta_batch = (1./2.)*torch.mm(img_out_batch, text_out.t())
        hash_matrix_batch = hash_matrix[data_ids,:]
        sim_matrix_batch = sim_matrix[data_ids,:]

        sim_sum = -torch.sum(theta_batch*sim_matrix_batch - torch.log(1. + torch.exp(theta_batch)))
        preserve_sim = torch.sum(torch.pow(hash_matrix_batch - img_out_batch, 2))
        preserve_balance = torch.sum(torch.pow(img_out_batch.t().mm(ones) + img_out[unupdated_ids].t().mm(ones_), 2))
        loss = sim_sum + gamma*preserve_sim + eta*preserve_balance
        loss /= (batch_size*ntrains)

        loss.backward()
        img_optimizer.step()
        loss_mean += loss.item()
    print('epoch: ', epoch, 'loss: ', loss_mean/len(train_loader))
    return img_out

def train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device, train_loader, text_optimizer, img_out, text_out, gamma, eta, ntrains):
    #img_model.train()
    #text_model.eval()

    loss_mean = 0
    for batch_idx, (data_ids, data_idxs, imgs, desc_vecs, tag_vecs) in enumerate(train_loader):
        batch_size = len(data_ids)
        imgs, desc_vecs, tag_vecs = imgs.to(device), desc_vecs.to(device), tag_vecs.to(device)
        unupdated_ids = np.setdiff1d(range(ntrains), data_ids)
        ones = torch.ones(batch_size, 1).to(device)
        ones_ = torch.ones(ntrains - batch_size, 1).to(device)

        text_optimizer.zero_grad()
        text_out_batch = text_model(desc_vecs)
        #print(text_out_batch)

        text_out[data_ids,:] = text_out_batch.data

        theta_batch = (1./2.)*torch.mm(text_out_batch, img_out.t())
        hash_matrix_batch = hash_matrix[data_ids,:]
        sim_matrix_batch = sim_matrix[data_ids,:]

        sim_sum = -torch.sum(theta_batch*sim_matrix_batch - torch.log(1. + torch.exp(theta_batch)))
        #print('sum', sim_sum.item())
        preserve_sim = torch.sum(torch.pow(hash_matrix_batch - text_out_batch, 2))
        #print('sim', preserve_sim.item())
        preserve_balance = torch.sum(torch.pow(text_out_batch.t().mm(ones) + text_out[unupdated_ids].t().mm(ones_), 2))
        #print('balance', preserve_balance.item())
        loss = sim_sum + gamma*preserve_sim + eta*preserve_balance
        loss /= (batch_size*ntrains)

        #torch.autograd.set_detect_anomaly(True)
        loss.backward()
        text_optimizer.step()
        #print(text_model.fc1.weight)
        #print(text_model.fc2.weight)
        loss_mean += loss.item()
    print('epoch: ', epoch, 'loss: ', loss_mean/len(train_loader))
    return text_out

def generate_img_code(img_model, query):
    model_out =  img_model(query)
    hash_code = torch.sign(model_out)
    return hash_code

def generate_text_code(text_model, query):
    model_out =  text_model(query)
    hash_code = torch.sign(model_out)
    return hash_code

def calc_sim_matrix(source_label, target_label, device):
    sim_matrix = (torch.mm(source_label, target_label.t()) > 0).type(torch.FloatTensor).to(device)
    return sim_matrix

def calc_loss(sim_matrix, hash_matrix, img_out, text_out, gamma, eta, ntrains, device):
    ones = torch.ones(ntrains, 1).to(device)

    theta = (1./2.)*torch.mm(img_out, text_out.t())
    sim_sum = -torch.sum(theta*sim_matrix - torch.log(1. + torch.exp(theta)))
    print('sum', sim_sum)
    preserve_sim = torch.sum(torch.pow(hash_matrix - img_out, 2)) + torch.sum(torch.pow(hash_matrix - text_out, 2))
    print('sim', preserve_sim)
    preserve_balance = torch.sum(torch.pow(img_out.t().mm(ones), 2)) + torch.sum(torch.pow(text_out.t().mm(ones), 2))
    print('balance', preserve_balance)

    loss = sim_sum + gamma*preserve_sim + eta*preserve_balance

    return loss.item()

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
    mAP /= len(queries)
    return mAP.item()

def train(img_model, text_model, source_data, vocab_size, device):

    nsource = len(source_data)
    # train validation split
    idx_list = [i for i in range(nsource)]
    val_idx_list = np.random.choice(idx_list, int(nsource/10), replace=False)
    train_idx_list = np.setdiff1d(idx_list, val_idx_list)
    train_data = [ [idx] + source_data[data_idx] for idx, data_idx in enumerate(train_idx_list) ] 
    val_data = [ [idx] + source_data[data_idx] for idx, data_idx in enumerate(val_idx_list) ] 

    lr_schedule = np.linspace(0.01, np.power(10, -6.), NEPOCH)

    ntrain = len(train_data)
    print(ntrain)
    train_loader_ = torch.utils.data.DataLoader(train_data, batch_size=ntrain)
    for batch_idx, (data_ids, data_idxs, imgs, doc_vecs, tag_vecs) in enumerate(train_loader_):
        label_set = tag_vecs
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    gamma = 1.
    eta = 1.

    img_out = torch.randn([ntrain, HASH_CODR_LENGTH]).to(device)
    text_out = torch.randn([ntrain, HASH_CODR_LENGTH]).to(device)

    hash_matrix = torch.sign(img_out + text_out)
    sim_matrix = calc_sim_matrix(label_set, label_set, device)

    img_optimizer = optim.SGD(img_model.parameters(), lr=0.01)
    text_optimizer = optim.SGD(text_model.parameters(), lr=0.01)

    loss_hist = []
    i2t_mAP_hist = []
    t2i_mAP_hist = [] 

    for epoch in range(NEPOCH):
        print('epoch: ', (epoch))
        img_out = train_cnn(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, img_optimizer, img_out, text_out, gamma, eta, ntrain)
        text_out = train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, text_optimizer, img_out, text_out, gamma, eta, ntrain)
        hash_matrix = torch.sign(img_out + text_out)
        loss = calc_loss(sim_matrix, hash_matrix, img_out, text_out, gamma, eta, ntrain, device)
        loss_hist.append(loss)
        print('loss: ', loss)
        i2t_mAP, t2i_mAP = validation(img_model, text_model, val_data, device)
        i2t_mAP_hist.append(i2t_mAP)
        t2i_mAP_hist.append(t2i_mAP)
        print('text to image mAP: ', i2t_mAP, 'image to text mAP: ', t2i_mAP)

        lr = lr_schedule[epoch]
        for param in img_optimizer.param_groups:
            param['lr'] = lr

        for param in text_optimizer.param_groups:
            param['lr'] = lr

    fig = plt.figure()
    plt.plot(range(NEPOCH), loss_hist, marker='.', label='train loss')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.savefig('result/train_loss.png')

    fig = plt.figure()
    plt.plot(range(NEPOCH), i2t_mAP_hist, marker='.', label='image to text mAP')
    plt.plot(range(NEPOCH), t2i_mAP_hist, marker='*', label='text to image mAP')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.savefig('result/val_mAP.png')


def validation(img_model, text_model, val_data, device):
    nval = len(val_data)

    idx_list = [i for i in range(nval)]
    #query_idx_list = np.random.choice(idx_list, 20, replace=False)
    query_idx_list = idx_list[:20]
    ret_idx_list = idx_list
    query_data = [ val_data[idx] for idx in query_idx_list ]
    ret_data = [ val_data[idx] for idx in ret_idx_list ]

    query_loader = torch.utils.data.DataLoader(query_data, batch_size=len(query_data))
    for data_ids, data_idxs, imgs, doc_vecs, tag_vecs in query_loader:
        imgs, doc_vecs, tag_vecs = imgs.to(device), doc_vecs.to(device), tag_vecs.to(device)
        query_img_hash_code = generate_img_code(img_model, imgs)
        query_text_hash_code = generate_text_code(text_model, doc_vecs)
        query_labels = tag_vecs

    ret_loader = torch.utils.data.DataLoader(ret_data, batch_size=len(ret_data))
    for data_ids, data_idxs, imgs, doc_vecs, tag_vecs in query_loader:
        imgs, doc_vecs, tag_vecs = imgs.to(device), doc_vecs.to(device), tag_vecs.to(device)
        ret_img_hash_code = generate_img_code(img_model, imgs)
        ret_text_hash_code = generate_text_code(text_model, doc_vecs)
        ret_labels = tag_vecs

    return calc_map(query_img_hash_code, query_labels, ret_text_hash_code, ret_labels),\
            calc_map(query_text_hash_code, query_labels, ret_img_hash_code, ret_labels)

def test(model):
    model.eval()

def tokenizer2(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def main():
    #nltk.download('stopwords')
    en_stopwords = stopwords.words('english')
    device = torch.device("cuda" if GPU else "cpu")

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

    img_model = CNNModel(IMG_SIZE, HASH_CODR_LENGTH).to(device)
    text_model = TextModel(desc_vocab_size, HASH_CODR_LENGTH).to(device)

    train(img_model, text_model, train_data, tag_vocab_size, device)

    img_model.save('model/img_model.t7')
    text_model.save('model/text_model.t7')

if  __name__ == '__main__':
    main()

