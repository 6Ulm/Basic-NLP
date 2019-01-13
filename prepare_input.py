import re, string, json
from nltk.corpus import stopwords
from itertools import chain
from tqdm import tqdm

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

def clean_text(text, remove_stopwords, stop_words):
    text = text.lower()
    text = text.replace('\n', ' ').replace('-', ' ')
    text = ''.join(' ' if l in punct else l for l in text)
    text = re.sub(' +',' ',text)
    text = text.strip()
    tokens = text.split()
    tokens = [word if len(word) <=20 else word[:10] for word in tokens]
    
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    return tokens

def text_correction(corrector, words, corpus, remove_stopwords, stopwords):
    tokenized_corpus = [clean_text(corpus[i], remove_stopwords, stopwords) \
                        for i in tqdm(range(len(corpus)))]
    print('Done tokenization and cleaning')
    
    vocab = set(chain.from_iterable(tokenized_corpus))
    words_ = set(words)
    vocab_not_err = [word for word in vocab if word in words_]
    vocab_err = [word for word in vocab if word not in words_]
    vocab_err_correction = [corrector.correction(vocab_err[i]) \
                            for i in tqdm(range(len(vocab_err)))]
    print('Done correction')
    
    vocab_ = vocab_not_err + vocab_err
    vocab_correct = vocab_not_err + vocab_err_correction
    vocab_dict = dict(zip(vocab_, vocab_correct))
    tokenized_corpus_correction = [[vocab_dict[word] for word in tokenized_corpus[i]] \
                                    for i in tqdm(range(len(tokenized_corpus)))]
    print('Done correction of corpus')
    return tokenized_corpus_correction

def vectorize_corpus(tokenized_corpus, truncated, max_size):
    vocab_final = set(chain.from_iterable(tokenized_corpus))
    vocab2index = dict(zip(vocab_final, range(len(vocab_final))))
    vectorized_tokens = [[vocab2index[word] for word in tokenized_corpus[i]] \
                        for i in tqdm(range(len(tokenized_corpus)))]
    print('Done vectorization of corpus')
    
    if not truncated:
        max_size = max([len(text) for text in vectorized_tokens])
        
    # truncate too-long text
    vectorized_tokens = [text[:min(len(text), max_size)] for text in vectorized_tokens]
    # padding zeros so that all vectorized texts have the same length
    vectorized_tokens = [text + [0]*(max_size-len(text)) if len(text) < max_size \
                        else text for text in vectorized_tokens]
    
    with open('train.json', 'w') as file:
        json.dump(vectorized_tokens, file)
    with open('vocab2index.json', 'w') as file:
        json.dump(vocab2index, file)

    return (vectorized_tokens, vocab2index)