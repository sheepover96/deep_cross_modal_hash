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
BATCH_SIZE = 32
NEPOCH = 100


def train_cnn(epoch, img_model, text_model, hash_matrix, sim_matrix, device, train_loader, img_optimizer, img_out, text_out, gamma, eta, ntrains):
    ones = torch.ones(ntrains, requires_grad=True).to(device)
    img_model.train()
    text_model.eval()

    loss_mean = 0
    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in enumerate(train_loader):
        batch_size = len(data_ids)
        imgs, tag_vecs = imgs.to(device), tag_vecs.to(device)
        img_out_batch = img_model(imgs)

        img_optimizer.zero_grad()
        img_out[data_ids,:] = img_out_batch

        theta = (1./2.)*torch.matmul(img_out, text_out.t())
        sim_sum = torch.sum(theta[data_ids,:]*sim_matrix[data_ids,:] - torch.log(1. + torch.exp(theta[data_ids,:])))
        #preserve_sim = torch.sum(torch.pow(hash_matrix[data_idxs,:] - img_out, 2)) +\
        #    torch.sum(torch.pow(hash_matrix[data_idxs,:] - text_out, 2))
        #preserve_balance = torch.pow(img_out.matmul(ones), 2) +\
        #    torch.pow(text_out.matmul(ones), 2)
        preserve_sim = torch.sum(torch.pow(hash_matrix[data_ids,:] - img_out_batch, 2))
        preserve_balance = torch.sum(torch.pow(img_out.t().matmul(ones), 2))
        loss = -sim_sum + gamma*preserve_sim + eta*preserve_balance
        loss /= (batch_size*ntrains)
        loss.backward(retain_graph=True)
        img_optimizer.step()
        loss.detach()
        loss_mean += loss.item()
    print('epoch: ', epoch, 'loss: ', loss_mean/len(train_loader))

def train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device, train_loader, text_optimizer, img_out, text_out, gamma, eta, ntrains):
    ones = torch.ones(ntrains, requires_grad=True).to(device)
    img_model.train()
    text_model.eval()

    loss_mean = 0
    for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in enumerate(train_loader):
        batch_size = len(data_ids)
        imgs, tag_vecs = imgs.to(device), tag_vecs.to(device)

        text_optimizer.zero_grad()
        text_out_batch = text_model(tag_vecs) 

        text_out[data_ids,:] = text_out_batch

        theta = (1./2.)*torch.matmul(img_out, text_out.t())
        sim_sum = torch.sum(theta[data_ids,:]*sim_matrix[data_ids,:] - torch.log(1. + torch.exp(theta[data_ids,:])))
        preserve_sim = torch.sum(torch.pow(hash_matrix[data_ids,:] - text_out_batch, 2))
        preserve_balance = torch.sum(torch.pow(text_out.t().matmul(ones), 2))
        loss = -sim_sum + gamma*preserve_sim + eta*preserve_balance
        loss /= (batch_size*ntrains)
        if batch_idx == 0:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        text_optimizer.step()
        loss.detach()
        loss_mean += loss.item()
    print('epoch: ', epoch, 'loss: ', loss_mean/len(train_loader))

def calc_loss(sim_matrix, hash_matrix, img_out, text_out, gamma, eta, ntrains, device):
    ones = torch.ones(ntrains).to(device)

    theta = (1./2.)*torch.matmul(img_out, text_out.t())
    sim_sum = torch.sum(theta*sim_matrix - torch.log(1. + torch.exp(theta)))
    preserve_sim = torch.sum(torch.pow(hash_matrix - img_out, 2)) + torch.sum(torch.pow(hash_matrix - text_out, 2))
    preserve_balance = torch.sum(torch.pow(img_out.t().matmul(ones), 2)) + torch.sum(torch.pow(text_out.t().matmul(ones), 2))

    loss = -sim_sum + gamma*preserve_sim + eta*preserve_balance

    return loss

def train(img_model, text_model, train_data, vocab_size, device):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    ntrain = len(train_data)
    gamma = 0
    eta = 0
    hash_matrix = torch.randn([ntrain, HASH_CODR_LENGTH], requires_grad=True).to(device)
    sim_matrix = torch.randn([ntrain, ntrain], requires_grad=True).to(device)

    img_out = torch.randn([ntrain, HASH_CODR_LENGTH], requires_grad=True).to(device)
    text_out = torch.randn([ntrain, HASH_CODR_LENGTH], requires_grad=True).to(device)

    img_optimizer = optim.SGD(img_model.parameters(), lr=0.01)
    text_optimizer = optim.SGD(text_model.parameters(), lr=0.01)

    for epoch in range(NEPOCH):
        print('epoch: ', (epoch+1))
        train_cnn(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, img_optimizer, img_out, text_out, gamma, eta, ntrain)
        train_text(epoch, img_model, text_model, hash_matrix, sim_matrix, device,\
            train_loader, text_optimizer, img_out, text_out, gamma, eta, ntrain)
        hash_matrix = torch.sign(img_out + text_out)
        loss = calc_loss(sim_matrix, hash_matrix, img_out, text_out, gamma, eta, ntrain, device)
        print('loss: ', loss.item())

def validation(img_model, text_model, train_data):
    ntrain = len(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)
    img_hash_codes = torch.zeros(ntrain, HASH_CODR_LENGTH, dtype=torch.float)
    text_hash_codes = torch.zeros(ntrain, HASH_CODR_LENGTH, dtype=torch.float)
    with torch.no_grad():
        for batch_idx, (data_ids, data_idxs, imgs, tag_vecs) in enumerate(train_loader):
            img_out_batch = img_model(imgs)
            img_hash_codes[data_ids,:] = img_out_batch
            text_out_batch = img_model(imgs)
            text_hash_codes[data_ids,:] = text_out_batch
        img_hash_codes = torch.sign(img_hash_codes)
        text_hash_codes = torch.sign(text_hash_codes)


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

    img_model = CNNModel(IMG_SIZE, HASH_CODR_LENGTH).to(device)
    text_model = TextModel(vocab_size, HASH_CODR_LENGTH).to(device)

    train(img_model, text_model, train_data, vocab_size, device)




if  __name__ == '__main__':
    main()
