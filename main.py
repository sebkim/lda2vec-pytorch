import numpy as np
import sys
import os
from utils import preprocess as pp
import torch
from training import train
from prepare import prepare, N_TOPICS
import argparse

EMBEDDING_DIM = 150

def parse_args():
    desc = "Pytorch implementation of lda2vec"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--n_epochs', type=int, default=100, help='The number of training n_epochs')
    parser.add_argument('--batch_size', type=int, default=1024*4, help='The size of batch size')

    parser.add_argument('--lambda_const', type=float, default=100.0, help='Strength of dirichlet prior.')
    parser.add_argument('--num_sampled', type=int, default=15, help='Number of negative words to sample.')
    parser.add_argument('--topics_weight_decay', type=float, default=1e-2, help='L2 regularization for topic vectors.')
    parser.add_argument('--topics_lr', type=float, default=1e-3, help='Learning rate for topic vectors.')
    parser.add_argument('--doc_weights_lr', type=float, default=1e-3, help='Learning rate for document weights.')
    parser.add_argument('--word_vecs_lr', type=float, default=1e-3, help='Learning rate for word vectors.')

    parser.add_argument('--save_every', type=int, default=10, help='Save the model from time to time.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradients by absolute value.')

    parser.add_argument('--device', type=str, default='cuda:0', help='Set gpu mode; [cpu, cuda:0, cuda:1, ...]')
    parser.add_argument('--num_workers', type=int, default='4', help='DataLoader num_workers')
    
    return parser.parse_args()

def load():
    unigram_distribution = np.load('./npy/unigram_distribution.npy', allow_pickle = True)
    decoder = np.load('./npy/decoder.npy', allow_pickle = True)
    data = np.load('./npy/data.npy', allow_pickle = True)
    doc_weights_init = np.load('./npy/doc_weights_init.npy', allow_pickle = True)
    return unigram_distribution, decoder, data, doc_weights_init

def main():
    os.makedirs('npy', exist_ok = True)
    os.makedirs('checkpoint', exist_ok = True)

    try:
        unigram_distribution, decoder, data, doc_weights_init = load()
    except:
        print(f"Required preprocess not done! Wait till preprocess done! ")
        prepare()
        unigram_distribution, decoder, data, doc_weights_init = load()
        print("Preprocess done!")
        print("")

    args = parse_args()

    decoder = decoder.item()
    word_vectors = np.random.normal(0, 0.01, (len(decoder), EMBEDDING_DIM))
    word_vectors = torch.FloatTensor(word_vectors).to(args.device)
    
    train(
        args, data, unigram_distribution, word_vectors,
        doc_weights_init = doc_weights_init,
        n_topics = N_TOPICS,
    )

if __name__ == '__main__':
    main()