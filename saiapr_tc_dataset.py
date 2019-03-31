import numpy as np
import pandas as pd
import spacy

import sys, os, csv, codecs
import ast
from PIL import Image
import xml.etree.ElementTree as ET

nlp = spacy.load('en')

DATASET_DEIR_PATH = './iapr_tc'
TRAIN_OUTPUT_DATASET_PATH = 'train_saiapr.csv'
TRAIN_OUTPUT_DATASET_PATH2 = 'train_saiapr2.csv'
TEST_OUTPUT_DATASET_PATH = 'test_saiapr.csv'
TEST_OUTPUT_DATASET_PATH2 = 'test_saiapr2.csv'
OUTPUT_TAGS_PATH = 'iapr_tags.txt'
OUTPUT_DOC_PATH = 'iapr_docs.txt'
TEST_DATA_NUM = 1000


def create_kv_dataset():
    tag_dict = {}
    with open(DATASET_DEIR_PATH + '/wlist.txt', 'r') as wlist:
        for line in wlist.read().splitlines():
            elements = line.split('\t')
            tag_dict[int(elements[0])] = elements[1]

    files = os.listdir(DATASET_DEIR_PATH + '/saiapr_tc-12')
    data_list = [[] for i in range(50000)]
    for file_path in files:
        if file_path != '.DS_Store':
            with open(DATASET_DEIR_PATH + '/saiapr_tc-12/' + file_path + '/labels.txt', 'r') as f:
                for tag in f.read().splitlines():
                    elements = tag.split()
                    if len(elements) > 0:
                        #print(data_list[int(elements[0])])
                        #print(tag_dict[int(elements[2])])
                        data_list[int(elements[0])].append(tag_dict[int(elements[2])])

    source_data_list = []
    img_data_no = 0
    for file_path in files:
        if file_path != '.DS_Store':
            img_paths = os.listdir(DATASET_DEIR_PATH + '/saiapr_tc-12/' + file_path + '/images')
            for img_path in img_paths:
                img_id = int(img_path[:-4])
                img_tags = data_list[img_id]
                full_img_path = DATASET_DEIR_PATH + '/saiapr_tc-12/' + file_path + '/images/' + img_path
                img = np.asarray(Image.open(full_img_path))
                if len(img.shape) == 3:
                    source_data_list.append([img_id, full_img_path, img_tags])
                    img_data_no += 1
                else:
                    print(img.shape)
    print(source_data_list)


    with open(TRAIN_OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as tr_wr,\
        open(TEST_OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as test_wr:
        tr_writer = csv.writer(tr_wr, lineterminator='\n')
        test_writer = csv.writer(test_wr, lineterminator='\n')
        idx_list = [i for i in range(len(source_data_list))]
        test_data_idxs = np.random.choice(idx_list, TEST_DATA_NUM, replace=False)
        train_data_idxs = np.setdiff1d(idx_list, test_data_idxs)
        for idx, train_idx in enumerate(train_data_idxs):
            element = source_data_list[train_idx]
            tr_writer.writerow(element)
        for idx, test_idx in enumerate(test_data_idxs):
            element = source_data_list[test_idx]
            test_writer.writerow(element)

def add_caption():
    train_df = pd.read_csv(TRAIN_OUTPUT_DATASET_PATH, header=None)
    test_df = pd.read_csv(TEST_OUTPUT_DATASET_PATH, header=None)
    train_data_list = []
    test_data_list = []
    files = os.listdir(DATASET_DEIR_PATH + '/annotations_complete_eng')
    for file_path in files:
        if file_path != '.DS_Store':
            annotations = os.listdir(DATASET_DEIR_PATH + '/annotations_complete_eng/' + file_path)
            for annotation_path in annotations:
                if 'eng' in annotation_path:
                    with codecs.open(DATASET_DEIR_PATH + '/annotations_complete_eng/' + file_path + '/' + annotation_path, 'r', encoding='utf-8', errors='ignore') as anf:
                        an_contents = anf.read()
                        an_dict = ET.fromstring(an_contents)
                        data_no = annotation_path[:-4]
                        element = train_df[train_df[0] == int(data_no)]
                        if not element.empty:
                            for desc in an_dict.iter('DESCRIPTION'):
                                des = desc
                            element = element.values[0]
                            img_no = element[0]
                            img_path = element[1]
                            labels = element[2]
                            train_data_list.append([img_no, img_path, des.text, labels])

                        element = test_df[train_df[0] == int(data_no)]
                        if not element.empty:
                            for desc in an_dict.iter('DESCRIPTION'):
                                des = desc
                            element = element.values[0]
                            img_no = element[0]
                            img_path = element[1]
                            labels = element[2]
                            test_data_list.append([img_no, img_path, des.text, labels])

    with open(TRAIN_OUTPUT_DATASET_PATH2, 'w', newline='', encoding='utf-8') as tr_wr,\
        open(TEST_OUTPUT_DATASET_PATH2, 'w', newline='', encoding='utf-8') as test_wr:
        tr_writer = csv.writer(tr_wr, lineterminator='\n')
        test_writer = csv.writer(test_wr, lineterminator='\n')
        for train_data in train_data_list:
            tr_writer.writerow(train_data)
        for test_data in test_data_list:
            test_writer.writerow(test_data)

def create_tags():
    df = pd.read_csv(TRAIN_OUTPUT_DATASET_PATH, header=None)
    tags = df[3]
    with open(OUTPUT_TAGS_PATH, 'w', encoding='utf-8') as wr2:
        for tag_list_str in tags:
            tag_list = ast.literal_eval(tag_list_str)
            for tag in tag_list:
                wr2.writelines(str(tag) + '\n')

def create_docs():
    df = pd.read_csv(TRAIN_OUTPUT_DATASET_PATH2, header=None)
    descriptions = df[2]
    with open(OUTPUT_DOC_PATH, 'w', encoding='utf-8') as wr2:
        for desc in descriptions:
            wr2.writelines(str(desc) + '\n')

    #df = pd.read_csv(TEST_OUTPUT_DATASET_PATH2, header=None)
    #descriptions = df[2]
    #with open(OUTPUT_DOC_PATH, 'w', encoding='utf-8') as wr2:
    #    for desc in descriptions:
    #        wr2.writelines(str(desc) + '\n')

if __name__ == '__main__':
    #create_kv_dataset()
    #create_tags()
    #add_caption()
    create_docs()