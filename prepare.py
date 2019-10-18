import numpy as np
import json
from utils import preprocess as pp
from tqdm import tqdm
from gensim import corpora, models

N_TOPICS = 20

def get_train_data(encoded_docs):
    data = []
    # new ids are created here
    for index, (_, doc) in tqdm(enumerate(encoded_docs)):
        windows = pp.get_windows(doc)
        # index represents id of a document, 
        # windows is a list of (word, window around this word),
        # where word is in the document
        data += [[index, w[0]] + w[1] for w in windows]
    data = np.array(data, dtype='int32')
    return data

def prepare():
    with open('data/newsgroups_texts.json', 'r') as fp:
        texts = json.load(fp)

    encoded_docs, decoder, word_counts = pp.preprocess(texts)
    word_counts = np.array(word_counts)
    unigram_distribution = word_counts/sum(word_counts)
    data = get_train_data(encoded_docs)

    
    np.save('./npy/unigram_distribution', unigram_distribution)
    np.save('./npy/data', data)
    np.save('./npy/decoder', decoder)
    print(f"unigram_distribution, data, and decoder saved!")

    # get LDA
    print("preprocess LDA starts...")
    htexts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
    dictionary = corpora.Dictionary(htexts)
    corpus = [dictionary.doc2bow(text) for text in htexts]

    lda = models.LdaModel(corpus, alpha='auto', id2word=dictionary, num_topics=N_TOPICS, passes=20)
    corpus_lda = lda[corpus]
    doc_weights_init = np.zeros((len(corpus_lda), N_TOPICS))
    for i in tqdm(range(len(corpus_lda))):
        topics = corpus_lda[i]
        for j, prob in topics:
            doc_weights_init[i, j] = prob
    np.save('npy/doc_weights_init', doc_weights_init)
    print("preprocess LDA done! doc_weights_init saved!")

if __name__ == '__main__':
    prepare()