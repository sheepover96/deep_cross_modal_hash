import pandas as pd

import sys, os, csv

DATASET_DEIR_PATH = './mirflickr/tags/'
OUTPUT_DATASET_PATH = 'kv_dataset.csv'

if __name__ == '__main__':
    files = os.listdir(DATASET_DEIR_PATH)
    with open(OUTPUT_DATASET_PATH, 'w', newline='', encoding='utf-8') as wr:
        writer = csv.writer(wr, lineterminator='\n')
        idx = 0
        for file_path in files:
            img_path = file_path.replace('tags','./mirflickr/im').replace('txt', 'jpg')
            with open(DATASET_DEIR_PATH + file_path, 'r') as f:
                for tag in f.read().splitlines():
                    if tag:
                        writer.writerow([str(idx), img_path, tag])
                        idx += 1