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
    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in enumerate(train_loader):
        batch_size = len(data_ids)
        imgs, tag_vecs = imgs.to(device), tag_vecs.to(device)
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
    ones = torch.ones(ntrains, 1).to(device)
    #img_model.train()
    #text_model.eval()

    loss_mean = 0
    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in enumerate(train_loader):
        batch_size = len(data_ids)
        imgs, tag_vecs = imgs.to(device), tag_vecs.to(device)
        unupdated_ids = np.setdiff1d(range(ntrains), data_ids)
        ones = torch.ones(batch_size, 1).to(device)
        ones_ = torch.ones(ntrains - batch_size, 1).to(device)

        text_optimizer.zero_grad()
        text_out_batch = text_model(tag_vecs)
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
        if torch.isnan(loss):
            break

        torch.autograd.set_detect_anomaly(True)
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

    return loss

def calc_map(queries, queries_label, target_item, target_label, k=None):
    target_out = model(target_item)
    target_hash_code = torch.sign(target_out)
    mAP = 0.
    for query_item, query_label in queries:
        query_out = model(query_item)
        query_hash_code = torch.sign(query_out)

        is_correct_target = (torch.matmul(query_label, target_label.t()) > 0).type(torch.FloatTensor)
        correct_target_num = is_correct_target.sum()
        if correct_target_num == 0:
            continue

        hamming_dist = torch.mm(query_hash_code, target_hash_code.t())
        _, hd_sorted_idx = hamming_dist.sort(descending=True)
        query_result = is_correct_target[hd_sorted_idx]

        count = torch.linspace(1, correct_target_num, correct_target_num)
        tindex = torch.nonzero(query_result == 1) + 1.
        mAP += count/tindex
    return mAP/len(queries)

def train(img_model, text_model, source_data, vocab_size, device):
    for (data_ids, data_idxs, imgs, tag_vecs) in train_data:
        if imgs.shape != (3, 256, 256):
            print(imgs.shape)
            print(data_ids)
            print(data_idxs)

    nsource = len(source_data)
    # train validation split
    idx_list = [i for i in range(nsource)]
    val_idx_list = np.random.choice(idx_list, int(nsource/10), replace=False)
    train_idx_list = np.setdiff1d(idx_list, val_idx_list)
    train_data = [ source_data[idx] for idx in train_idx_list ] 
    val_data = [ source_data[idx] for idx in val_idx_list ] 

    ntrain = len(train_data)
    print(ntrain)
    train_loader_ = torch.utils.data.DataLoader(train_data, batch_size=ntrain)
    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in enumerate(train_loader_):
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

    for epoch in range(NEPOCH):
        print('epoch: ', (epoch))
        img_out = train_cnn(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, img_optimizer, img_out, text_out, gamma, eta, ntrain)
        text_out = train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, text_optimizer, img_out, text_out, gamma, eta, ntrain)
        hash_matrix = torch.sign(img_out + text_out)
        loss = calc_loss(sim_matrix, hash_matrix, img_out, text_out, gamma, eta, ntrain, device)
        print('loss: ', loss.item())

def validation(img_model, text_model, val_data):
    nval = len(val_data)

    idx_list = [i for i in range(nval)]
    query_idx_list = np.random.choice(idx_list, 20, replace=False)
    ret_idx_list = np.setdiff1d(idx_list, query_idx_list)
    query_data = [ val_data[idx] for idx in query_idx_list ] 
    ret_data = [ val_data[idx] for idx in ret_idx_list ] 

    query_loader = torch.utils.data.DataLoader(query_data, batch_size=len(query_data))
    for data_ids, data_idxs, imgs, tag_vecs in enumerate(query_loader):
        query_img_hash_code = generate_img_code(img_model, imgs)
        query_text_hash_code = generate_text_code(text_model, tag_vecs)
        query_labels = tag_vecs

    ret_loader = torch.utils.data.DataLoader(ret_data, batch_size=len(ret_data))
    for data_ids, data_idxs, imgs, tag_vecs in enumerate(query_loader):
        ret_img_hash_code = generate_img_code(img_model, imgs)
        ret_text_hash_code = generate_text_code(text_model, tag_vecs)
        ret_labels = tag_vecs

    calc_map(query_img_hash_code, query_labels, ret_text_hash_code, ret_labels)
    calc_map(query_text_hash_code, query_labels, ret_img_hash_code, ret_labels)




def test(model):
    model.eval()

def tokenizer2(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def main():
    device = torch.device("cuda" if GPU else "cpu")

    TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True, stop_words=['<eos>'])
    lang = LanguageModelingDataset(path='./iapr_tags.txt', text_field=TEXT)
    TEXT.build_vocab(lang)
    vocab = TEXT.vocab
    vocab_size = len(vocab.freqs) + 1
    #print(vocab.itos)
    #print(vocab.stoi)

    train_data = DcmhDataset('./train_saiapr.csv', vocab.stoi, vocab_size)
    test_data = DcmhDataset('./test_saiapr.csv', vocab.stoi, vocab_size)

    img_model = CNNModel(IMG_SIZE, HASH_CODR_LENGTH).to(device)
    text_model = TextModel(vocab_size, HASH_CODR_LENGTH).to(device)

    train(img_model, text_model, train_data, vocab_size, device)



if  __name__ == '__main__':
    main()
