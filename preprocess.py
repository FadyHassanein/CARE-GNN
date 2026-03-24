"""
Preprocess .mat data files into adjacency list pickle files required by CARE-GNN.

Usage:
    python preprocess.py          # Process both datasets
    python preprocess.py yelp     # Process only Yelp
    python preprocess.py amazon   # Process only Amazon
"""
import os
import sys
import pickle
from collections import defaultdict
from scipy.io import loadmat
import scipy.sparse as sp


def sparse_to_adjlist(sp_matrix, filename):
    """Convert sparse matrix to adjacency list and save as pickle."""
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    print(f'  Saved {filename} ({len(adj_lists)} nodes)')


def process_yelp(prefix='data/'):
    mat_path = prefix + 'YelpChi.mat'
    if not os.path.exists(mat_path):
        print(f'Error: {mat_path} not found. Extract YelpChi.zip first.')
        return False
    print('Processing YelpChi...')
    data = loadmat(mat_path)
    sparse_to_adjlist(data['homo'], prefix + 'yelp_homo_adjlists.pickle')
    sparse_to_adjlist(data['net_rur'], prefix + 'yelp_rur_adjlists.pickle')
    sparse_to_adjlist(data['net_rtr'], prefix + 'yelp_rtr_adjlists.pickle')
    sparse_to_adjlist(data['net_rsr'], prefix + 'yelp_rsr_adjlists.pickle')
    print('YelpChi done.\n')
    return True


def process_amazon(prefix='data/'):
    mat_path = prefix + 'Amazon.mat'
    if not os.path.exists(mat_path):
        print(f'Error: {mat_path} not found. Extract Amazon.zip first.')
        return False
    print('Processing Amazon...')
    data = loadmat(mat_path)
    sparse_to_adjlist(data['homo'], prefix + 'amz_homo_adjlists.pickle')
    sparse_to_adjlist(data['net_upu'], prefix + 'amz_upu_adjlists.pickle')
    sparse_to_adjlist(data['net_usu'], prefix + 'amz_usu_adjlists.pickle')
    sparse_to_adjlist(data['net_uvu'], prefix + 'amz_uvu_adjlists.pickle')
    print('Amazon done.\n')
    return True


if __name__ == '__main__':
    dataset = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'

    if dataset in ('all', 'yelp'):
        process_yelp()
    if dataset in ('all', 'amazon'):
        process_amazon()
