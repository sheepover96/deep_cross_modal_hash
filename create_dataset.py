import pandas as pd
import spacy

import sys, os, csv
import ast

nlp = spacy.load('en')

DATASET_DEIR_PATH = './mirflickr/tags/'
TRAIN_OUTPUT_DATASET_PATH = 'train_dataset.csv'
TEST_OUTPUT_DATASET_PATH = 'test_dataset.csv'
OUTPUT_TAGS_PATH = 'tags.txt'


def create_kv_dataset():
    files = os.listdir(DATASET_DEIR_PATH)
    with open(TRAIN_OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as tr_wr,\
        open(TEST_OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as test_wr:
        tr_writer = csv.writer(tr_wr, lineterminator='\n')
        test_writer = csv.writer(test_wr, lineterminator='\n')
        train_idx = 0
        test_idx = 0
        for file_path in files:
            img_no = file_path.replace('tags','').replace('.txt', '')
            img_path = file_path.replace('tags','./mirflickr/im').replace('txt', 'jpg')
            with open(DATASET_DEIR_PATH + file_path, 'r') as f:
                tag_list = []
                for tag in f.read().splitlines():
                    if tag and tag in nlp.vocab:
                        tag_list.append(tag)
                if len(tag_list) > 19:
                    tr_writer.writerow([str(train_idx), img_no, img_path, tag_list])
                    train_idx += 1
                else:
                    test_writer.writerow([str(test_idx), img_no, img_path, tag_list])
                    test_idx += 1


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