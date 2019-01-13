# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
# simplified and fast spell correction

import os
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.word2vec import Word2Vec

###########################################
### Fast spelling (faster and more flexible than Textblob and Pyspellchecking)
### Allow to use dictionary imported from different pretrained models:
### GoogleNews, Glove, Twitter, Wiki, etc
### Supported formats: .vec, .txt, .bin.gz
###########################################

class FastSpelling():
    def __init__(self, input_file):
        
        # convert to gensim file
        if input_file.endswith('.txt'):
            output_text = input_file + '.gensim'
            if not os.path.isfile(output_text):
                print('generating .gensim file ...')
                glove2word2vec(glove_input_file=input_file, word2vec_output_file= output_text)        
            print('loading vocabulary of pretrained model ...')
            self.model = KeyedVectors.load_word2vec_format(output_text)

        elif input_file.endswith('.vec'):
            # convert to '.bin' file to reduce file size
            output_text = input_file + '.bin'
            if not os.path.isfile(output_text):
                print('generating .bin file ...')
                vec_model = KeyedVectors.load_word2vec_format(input_file, binary=False)
                vec_model.save_word2vec_format(output_text, binary=True)
            print('loading vocabulary of pretrained model ...')
            self.model = KeyedVectors.load_word2vec_format(output_text, binary=True)

        elif input_file.endswith('.bin.gz'):
            output_text = input_file
            print('loading vocabulary of pretrained model ...')
            self.model = KeyedVectors.load_word2vec_format(output_text, binary=True)

        # build vocabulary         
        self.words = self.model.index2word
        self.WORDS = dict((word, i) for i,word in enumerate(self.words))
        print('Vocabulary built')

    # Spelling correction engine
    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.WORDS.get(word, 0)

    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) \
                or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

###############################################
#### LOAD EMBEDDING MATRIX
#### Supported formats: .vec, .txt, .bin.gz
#### Allows to load embedding matrix of:
#### GoogleNews, GLOVE, Twitter, Wiki
###############################################

def get_embedding(pretrained_model, vocab2index):
    initial_word_vector_dim = pretrained_model['init_dimension']
    input_file = pretrained_model['file']

    word_vec = Word2Vec(size=initial_word_vector_dim, min_count=1)
    word_vec.build_vocab([[word] for word in vocab2index.keys()])

    # read input file using gensim
    if input_file.endswith('.txt'):
        output_text = input_file + '.gensim'
        if not os.path.isfile(output_text):
            print('generating .gensim file ...')
            glove2word2vec(glove_input_file=input_file, word2vec_output_file= output_text)
        print('matching vocabulary with pretrained vocabulary ...')
        word_vec.intersect_word2vec_format(output_text, binary=False, lockf=1.0)
        
    elif input_file.endswith('.vec'):
        # convert to '.bin' file to reduce file size
        vec_bin_file = input_file + '.bin'
        if not os.path.isfile(vec_bin_file):
            print('generating .bin file ...')
            vec_model = KeyedVectors.load_word2vec_format(input_file, binary=False)
            vec_model.save_word2vec_format(vec_bin_file, binary=True)
        print('matching vocabulary with pretrained vocabulary ...')
        word_vec.intersect_word2vec_format(vec_bin_file, binary=True)
    
    elif input_file.endswith('.bin.gz'):
        vec_bin_file = input_file
        print('matching vocabulary with pretrained vocabulary ...')
        word_vec.intersect_word2vec_format(vec_bin_file, binary=True)
    
    # load embedding matrix
    embedding = word_vec.wv.syn0
    # add zero vector (for padding special token)
    pad_vec = np.zeros((1,initial_word_vector_dim))
    embedding = np.insert(embedding, 0,pad_vec,0)
    # add Gaussian initialized vector (for OOV special token)
    oov_vec = np.random.normal(size= initial_word_vector_dim) 
    embedding = np.insert(embedding,0,oov_vec,0)
    print('embedding loaded')
    return embedding