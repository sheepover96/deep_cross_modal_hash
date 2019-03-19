import pandas as pd
import spacy

import sys, os, csv
import ast

nlp = spacy.load('en')

DATASET_DEIR_PATH = './iapr_tc'
TRAIN_OUTPUT_DATASET_PATH = 'train_saiapr.csv'
TEST_OUTPUT_DATASET_PATH = 'test_saiapr.csv'
OUTPUT_TAGS_PATH = 'iapr_tags.txt'


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

    train_data_list = []
    test_data_list = []
    img_data_no = 0
    for file_path in files:
        if file_path != '.DS_Store':
            img_paths = os.listdir(DATASET_DEIR_PATH + '/saiapr_tc-12/' + file_path + '/images')
            for img_path in img_paths:
                img_id = int(img_path[:-4])
                img_tags = data_list[img_id]
                full_img_path = DATASET_DEIR_PATH + '/saiapr_tc-12/' + file_path + '/images/' + img_path
                train_data_list.append([img_data_no, img_id, full_img_path, img_tags])
                img_data_no += 1


    with open(TRAIN_OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as tr_wr,\
        open(TEST_OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as test_wr:
        tr_writer = csv.writer(tr_wr, lineterminator='\n')
        test_writer = csv.writer(test_wr, lineterminator='\n')
        for elements in train_data_list:
            tr_writer.writerow(elements)


def create_tags():
    df = pd.read_csv(TRAIN_OUTPUT_DATASET_PATH, header=None)
    tags = df[3]
    with open(OUTPUT_TAGS_PATH, 'w', encoding='utf-8') as wr2:
        for tag_list_str in tags:
            tag_list = ast.literal_eval(tag_list_str)
            for tag in tag_list:
                wr2.writelines(str(tag) + '\n')

if __name__ == '__main__':
    create_kv_dataset()
    create_tags()