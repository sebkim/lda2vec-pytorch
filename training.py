import numpy as np
import torch
import torch.optim as optim
import math
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from lda2vec_loss import loss, topic_embedding
from glob import glob

# negative sampling power
BETA = 0.75


def train(args, data, unigram_distribution, word_vectors, doc_weights_init, n_topics):
    """Trains a lda2vec model. Saves the trained model and logs.

    'data' consists of windows around words. Each row in 'data' contains:
    id of a document, id of a word, 'window_size' words around the word.

    Arguments:
        data: A numpy int array with shape [n_windows, window_size + 2].
        unigram_distribution: A numpy float array with shape [vocab_size].
        word_vectors: A numpy float array with shape [vocab_size, embedding_dim].
        doc_weights_init: A numpy float array with shape [n_documents, n_topics] or None.
        n_topics: An integer.
    """
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lambda_const = args.lambda_const
    num_sampled = args.num_sampled
    topics_weight_decay = args.topics_weight_decay
    topics_lr = args.topics_lr
    doc_weights_lr = args.doc_weights_lr
    word_vecs_lr = args.word_vecs_lr
    save_every = args.save_every
    grad_clip = args.grad_clip
    device = args.device
    num_workers = args.num_workers

    n_windows = len(data)
    n_documents = len(np.unique(data[:, 0]))
    vocab_size, embedding_dim = word_vectors.shape

    print('number of documents:', n_documents)
    print('number of windows:', n_windows)
    print('number of topics:', n_topics)
    print('vocabulary size:', vocab_size)
    print('word embedding dim:', embedding_dim)

    # prepare word distribution
    unigram_distribution = torch.FloatTensor(unigram_distribution**BETA)
    unigram_distribution /= unigram_distribution.sum()
    unigram_distribution = unigram_distribution.to(device)

    # create a data feeder
    dataset = SimpleDataset(torch.IntTensor(data))
    iterator = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=True, drop_last=True
    )

    # create a lda2vec model
    topics = topic_embedding(n_topics, embedding_dim)
    topics = topics.to(device)
    
    model = loss(
        device, topics, word_vectors, unigram_distribution,
        n_documents, lambda_const, num_sampled
    )
    model = model.to(device)
    model.doc_weights.weight.data = torch.FloatTensor(doc_weights_init).to(device)
    
    # # load checkpoint
    # model.load_state_dict(checkpoint)
    # ###

    params = [
        {'params': [model.topics.topic_vectors],
         'lr': topics_lr, 'weight_decay': topics_weight_decay},
        {'params': [model.doc_weights.weight],
         'lr': doc_weights_lr},
        {'params': [model.neg.embedding.weight],
         'lr': word_vecs_lr}
    ]
    optimizer = optim.Adam(params)
    n_batches = math.ceil(n_windows/batch_size)
    print('number of batches:', n_batches, '\n')
    losses = []  # collect all losses here

    start_epoch = 1
    # if checkpoint exists
    model_list = glob(os.path.join('checkpoint', '*.pt'))
    model_list2 = []
    if not len(model_list) == 0:
        for m in model_list:
            model_list2.append((m, int(m.split('/')[-1].split('_')[0])))
        model_list2 = sorted(model_list2, key=lambda x:x[1])
        
        start_epoch = model_list2[-1][1] + 1
        model.load_state_dict(torch.load(model_list2[-1][0]))
        print(f" [*] Load SUCCESS: {model_list2[-1][0]}")
    ###
    
    model.train()
    try:
        for epoch in range(start_epoch, n_epochs + 1):

            print('epoch', epoch)
            running_neg_loss = 0.0
            running_dirichlet_loss = 0.0

            for batch in tqdm(iterator):
                batch = batch.to(device)

                doc_indices = batch[:, 0]
                pivot_words = batch[:, 1]
                target_words = batch[:, 2:]

                neg_loss, dirichlet_loss = model(doc_indices, pivot_words, target_words)
                total_loss = neg_loss + dirichlet_loss

                optimizer.zero_grad()
                total_loss.backward()

                # gradient clipping
                for p in model.parameters():
                    p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

                optimizer.step()

                n_samples = batch.size(0)
                running_neg_loss += neg_loss.data.item()*n_samples
                running_dirichlet_loss += dirichlet_loss.data.item()*n_samples

            losses += [(epoch, running_neg_loss/n_windows, running_dirichlet_loss/n_windows)]
            print('neg_loss: {0:.2f}, dirichlet_loss: {1:.2f}'.format(*losses[-1][1:]))
            if epoch % save_every == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), 'checkpoint/' + str(epoch) + '_epoch_model_state.pt')
                _write_training_logs(losses)

    except (KeyboardInterrupt, SystemExit):
        print(' Interruption detected, exiting the program...')

    
    torch.save(model.state_dict(), 'model_state.pt')


def _write_training_logs(losses):
    with open('training_logs.txt', 'w') as f:
        column_names = 'epoch,negative_sampling_loss,dirichlet_prior_loss\n'
        f.write(column_names)
        for i in losses:
            values = ('{0},{1:.3f},{2:.3f}\n').format(*i)
            f.write(values)


class SimpleDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index].type(torch.LongTensor)

    def __len__(self):
        return self.data_tensor.size(0)
