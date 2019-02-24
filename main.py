import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchtext.data import Field
from torchtext.datasets import LanguageModelingDataset
import spacy
spacy_en = spacy.load('en')

from data_loader import DcmhDataset
from dcmh_model import CNNModel, TextModel


GPU = torch.cuda.is_available()

HASH_CODR_LENGTH = 100
IMG_SIZE = 256
BATCH_SIZE = 256
NEPOCH = 100


def train_cnn(epoch, img_model, bw_model, hash_matrix, sim_matrix, device, train_loader, img_optimizer, img_out, text_out, gamma, eta):
    ntrains = len(train_loader)
    ones = torch.ones(len(ntrains))
    img_model.train()
    bw_model.eval()

    img_out = torch.randn((len(ntrains), HASH_CODR_LENGTH))
    text_out = torch.randn((len(ntrains), HASH_CODR_LENGTH))

    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in train_loader:
        batch_size = len(data_ids)
        imgs, tag_vecs = imgs.to(device), tag_vecs.to(device)
        img_out_batch = img_model(imgs)
        text_out_batch = bw_model(tag_vecs) 

        img_out[data_idxs,:] = img_out_batch

        theta = (1./2.)*torch.matmul(img_out.t(), text_out)
        sim_sum = torch.sum(sim_matrix[data_idxs,:]*theta - torch.log(1. + torch.exp(theta)))
        #preserve_sim = torch.sum(torch.pow(hash_matrix[data_idxs,:] - img_out, 2)) +\
        #    torch.sum(torch.pow(hash_matrix[data_idxs,:] - text_out, 2))
        #preserve_balance = torch.pow(img_out.matmul(ones), 2) +\
        #    torch.pow(text_out.matmul(ones), 2)
        preserve_sim = torch.sum(torch.pow(hash_matrix[data_idxs,:] - img_out, 2))
        preserve_balance = torch.pow(img_out.matmul(ones), 2)
        loss = -sim_sum + gammma*preserve_sim + eta*preserve_balance
        img_optimizer.zero_grad()
        loss.backward()
        img_optimizer.step()


def train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device, train_loader, text_optimizer, img_out, text_out, gamma, eta):
    ntrains = len(train_loader)
    ones = torch.ones(ntrains)
    img_model.train()
    bw_model.eval()
    gammma = 0
    eta = 0


    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in train_loader:
        batch_size = len(data_ids)
        imgs, tag_vecs = imgs.to(device), tag_vecs.to(device)
        text_out_batch = text_model(tag_vecs) 

        text_out[data_ids,:] = text_out_batch

        theta = (1./2.)*torch.matmul(img_out.t(), text_out)
        sim_sum = torch.sum(sim_matrix*theta - torch.log(1. + torch.exp(theta)))
        preserve_sim = torch.sum(torch.pow(hash_matrix - text_outp, 2))
        preserve_balance = torch.pow(text_output.matmul(ones), 2)
        loss = -sim_sum + gammma*preserve_sim + eta*preserve_balance
        text_optimizer.zero_grad()
        loss.backward()
        text_optimizer.step()
    
def train(img_model, text_model, train_loader, vocab_size, device):
    ntrain = len(train_loader)
    gamma = 0
    eta = 0
    hash_matrix = torch.randn(HASH_CODR_LENGTH, ntrain)
    sim_matrix = torch.randn(ntrain, ntrain)

    img_out = torch.randn(ntrain, HASH_CODR_LENGTH)
    text_out = torch.randn((ntrain, HASH_CODR_LENGTH))

    img_model = CNNModel(IMG_SIZE, HASH_CODR_LENGTH)
    text_model = TextModel(vocab_size, HASH_CODR_LENGTH)

    img_optimizer = optim.SGD(img_model.parameters(), lr=0.001)
    text_optimizer = optim.SGD(text_model.parameters(), lr=0.001)

    for epoch in range(NEPOCH):
        print('epoch: ' + (epoch+1))
        train_cnn(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, img_optimizer, img_out, text_out, gamma, eta)
        train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, text_optimizer, img_out, text_out, gamma, eta)
        hash_matrix = torch.sign(img_out + text_out)
        


def test(model):
    model.eval()

def tokenizer2(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def main():
    device = torch.device("cuda" if GPU else "cpu")

    TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True)
    lang = LanguageModelingDataset(path='./tags.txt', text_field=TEXT)
    TEXT.build_vocab(lang, min_freq=3)
    vocab = TEXT.vocab
    vocab_size = len(vocab.freqs)
    #print(vocab.itos)
    #print(vocab.stoi)

    train_data = DcmhDataset('./train_dataset.csv', vocab.stoi, vocab_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))




if  __name__ == '__main__':
    main()
