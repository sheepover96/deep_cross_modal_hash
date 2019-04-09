import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import sys, os, csv, pickle

DATASET_DIR_PATH = './iapr_tc'



def main():
    directed_graph = nx.DiGraph()
    files = os.listdir(DATASET_DIR_PATH + '/saiapr_tc-12')
    for file_path in files:
        if file_path != '.DS_Store':
            df = pd.read_csv(os.path.join(DATASET_DIR_PATH, 'saiapr_tc-12', file_path, 'ontology_path.txt'), header=None, delimiter='\t')
            for row in df.iterrows():
                ontology = row[1][2][10:-2]
                words = ontology.split('->')
                cleaned_words = list(map(lambda word: word[1:] if word.startswith('_') else word, words))
                #print(cleaned_words)
                child_edge = [ (cleaned_words[idx], cleaned_words[idx+1]) for idx in range(len(cleaned_words)-1)]
                print(child_edge)
                parent_edge = [ (cleaned_words[idx], cleaned_words[idx-1]) for idx in reversed(range(len(cleaned_words)-1))]
                directed_graph.add_nodes_from(cleaned_words)
                directed_graph.add_edges_from(child_edge)
                directed_graph.add_edges_from(parent_edge)
    
    nx.draw_networkx(directed_graph)
    plt.show()
    with open('dataset/iapr_tc_wordnet.pickle', 'wb') as f:
        pickle.dump(directed_graph, f)


if __name__ == '__main__':
    main()