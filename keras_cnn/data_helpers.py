import numpy as np
import re
import itertools
from collections import Counter
import h5py
import pickle

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(pos_data, neg_data):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(pos_data, "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_data, "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_test_data_and_labels(test_file, test_label, sequence_length):
    x_text = list(open(test_file, "r", encoding='latin-1').readlines())
    y_text = x_text.pop().split(" ")
    x_text = [s.strip() for s in x_text]
    
    # Split by words
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    # Generate labels
    y = []
    for label in y_text:
        if int(label) == 0:
            y.append([0, 1]);
        else:
            y.append([1, 0]);

    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_embedding_vocab(embeddings):
    """
    Builds a vocabulary mapping from word to index based on the word embeddings.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    vocabulary_inv = list(sorted(embeddings.keys()))
    # Mapping from word to index
    return vocabulary_inv

def build_join_vocab(sentences, embeddings_file):
    data_vocab, data_vocab_inv = build_vocab(sentences)

    embeddings = load_embeddings(embeddings_file)
    embeddings_vocab_inv = build_embedding_vocab(embeddings)

    join_vocab_inv = list(set(data_vocab_inv + embeddings_vocab_inv))

    join_vocabulary = {x: i for i, x in enumerate(join_vocab_inv)}

    return [join_vocabulary, join_vocab_inv, ]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    # print(vocabulary)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data(pos_data, neg_data):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(pos_data, neg_data)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_test_data(test_file, test_label, sequence_length, vocabulary_file):
    sentences, labels = load_test_data_and_labels(test_file, test_label, sequence_length)
    sentences_padded = pad_test_sentences(sentences, sequence_length=sequence_length)
    vocabulary = load_vocabulary(vocabulary_file)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y]

def pad_test_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def load_vocabulary(vocabulary_file):
    pickle_in = open("vocabulary_file","rb")
    vocab = pickle.load(pickle_in)
    return vocab

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass 
    return False

def load_embeddings(word_embedding_file):
    embeddings_index = {}
    f = open(word_embedding_file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        if is_number(values[1]):
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Load embedding done: " + str(len(embeddings_index)))
    f.close()
    return embeddings_index

def get_embeddings(word_embedding_file, word_index, embedding_dim):
    embeddings_index = load_embeddings(word_embedding_file)
    print('--- start assigning embedding ---')
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    i=0
    # print(embeddings_index)
    print(len(word_index))
    print("embedding_matrix_size: " + str(len(embedding_matrix)))
    for word in word_index:
        embedding_vector = embeddings_index.get(word)
        # print("len(embedding vector): " + str(len(embedding_vector)))
        if embedding_vector is not None and len(embedding_vector)==embedding_dim:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            i=i+1
    return embedding_matrix

def load_data_embeddings_vocab(pos_data, neg_data, embeddings_file):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(pos_data, neg_data)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_join_vocab(sentences_padded, embeddings_file)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_test_data_separate_files(test_data_pos, test_data_neg, vocabulary, sequence_length):
    sentences, labels = load_data_and_labels(test_data_pos, test_data_neg)
    sentences_padded = pad_test_sentences(sentences, sequence_length)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y]

def string_to_input_data(sentence, label, vocabulary, sequence_length):
    sentences = [s.strip() for s in sentence]
    x_text = [clean_str(sent) for sent in sentences]
    x_text = [s.split(" ") for s in x_text]

    sentences_padded = pad_test_sentences(x_text, sequence_length)
    y = []

    if int(label) == 1:
        y.append([0, 1]);
    else:
        y.append([1, 0]);
        
    x, y = build_input_data(sentences_padded, y, vocabulary)
    return [x,y]