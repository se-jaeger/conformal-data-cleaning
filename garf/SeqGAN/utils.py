import linecache
import os
import random
import sqlite3
from pathlib import Path

import numpy as np
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical


class Vocab:  # Building a Vocabulary
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)  # Create a dictionary
        self.id2word = {v: k for k, v in self.word2id.items()}  # Reverse pass dictionary to id2word
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = (
                    word_counter.get(word, 0) + 1
                )  # The dictionary type assignment, where .get(word,0)+1 is a count of the frequency of word occurrences

        for word, count in sorted(
            word_counter.items(),
            key=lambda x: -x[1],
        ):  # The sorted() function sorts all iterable objects, sorting them from most frequent to least frequent.
            if count < min_count:
                break
            _id = len(self.word2id)  # Current dictionary size
            self.word2id.setdefault(
                word,
                _id,
            )  # Returns the value corresponding to the word in the dictionary, i.e. the number of occurrences of the word in the current sentence, or _id if it does not exist
            self.id2word[_id] = word

        self.raw_vocab = {
            w: word_counter[w] for w in self.word2id.keys() if w in word_counter
        }  # Dictionary Collection {each word: corresponding id}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]


def load_data(path, order):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM '{path}'")
    rows = cursor.fetchall()
    rows = [x[:-1] for x in rows]

    if order == 1:
        print("Load data in positive order...")

    elif order == 0:
        print("Loading data in reverse order...")
        rows = [x[::-1] for x in rows]

    cursor.close()
    conn.close()

    return rows


def sentence_to_ids(vocab, sentence, UNK=3):
    """# Arguments:
        vocab: SeqGAN.utils.Vocab
        sentence: list of str
    # Returns:
        ids: list of int.
    """
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids


def pad_seq(seq, max_length, PAD=0):  # If the length of the sentence is less than 25, 0 is added after it
    """:param seq: list of int,
    :param max_length: int,
    :return seq: list of int,
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq


def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    """:param ids: list of int,
    :param vocab:
    :param verbose(optional):
    :return sentence: list of str
    """
    sentence = []
    for i, id in enumerate(ids):
        vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(sentence)
    if verbose:
        print(sentence)
    return sentence


def addtwodimdict(thedict, key_a, key_b, val):
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a: {key_b: val}})


class GeneratorPretrainingGenerator(
    Sequence,
):  # Take the data directly from the original data and make x and y_true as x_train and y_train i.e. training data and labels
    def __init__(self, path, order, B, T, models_base_path, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<S>"
        self.EOS_TOKEN = "</S>"
        self.path = path
        self.B = B
        self.T = T
        self.models_base_path = models_base_path
        self.min_count = min_count
        self.count = 0

        sentences = load_data(path, order)

        # print("Raw data", sentences)
        self.rows = sentences

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)

        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        self.n_data = len(sentences)  # Number of original data rows
        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word

        with open(Path(self.models_base_path / "word2id.txt"), "w") as file:
            file.write(str(self.word2id))

        with open(Path(self.models_base_path / "id2word.txt"), "w") as file:
            file.write(str(self.id2word))

        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()

        self.reset()

    def __len__(self):
        return self.n_data // self.B  # Total number of data divided by the number of samples selected for one training

    def __getitem__(self, idx):  # Read the idx row of the original data, generate x and y_true
        """Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            x: numpy.array, shape = (B, max_length)
            y_true: numpy.array, shape = (B, max_length, V)
                labels with one-hot encoding.
                max_length is the max length of sequence in the batch.
                if length smaller than max_length, the data will be padded.
        """
        x, y_true = [], []
        start = (idx - 1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0
        for i in range(start, end):
            if self.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i

            sentence = self.rows[idx]  # Read the idx row of the query result
            words = []
            for i in sentence:
                words.append(i)
            ids = sentence_to_ids(
                self.vocab,
                words,
            )  # ids is a list that holds the sequence of word ids in the original data

            ids_x, ids_y_true = [], []  # place empty

            ids_x.append(self.BOS)  # Begin by writing the identifier BOS
            ids_x.extend(ids)  # Add ids, i.e., the ids corresponding to the multiple words in the current sentence
            ids_x.append(self.EOS)  # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            ids_y_true.extend(ids)
            ids_y_true.append(self.EOS)  # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)

            max_length = max(max_length, len(ids_x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(x):
            x[i] = x[i][
                :max_length
            ]  # loop len(X) times, X[i] is the i-th sentence in the list X, and truncated to the length of max_length

        for i, ids in enumerate(y_true):
            y_true[i] = y_true[i][:max_length]

        x = [
            pad_seq(sen, max_length) for sen in x
        ]  # If the length of the sentence is less than 25, 0 is added after it
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)
        # print("y_true:", y_true[0][0])
        y_true = to_categorical(
            y_true,
            num_classes=self.V,
        )  # The original category vector is converted to one-hot form, and the dimension is the total number of words.
        # print("x:", x[0])
        # print("y_true after conversion:",y_true[0][0])
        # print("x:", x)
        # print("y_true:",y_true)

        return (x, y_true)

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return (x, y_true)

    def reset(self):  # Reset to regenerate a jumbled array of size n_data
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)  # Break up the elements in the list
        # print(self.shuffled_indices)                       #A scrambled array of size n_data, [3850 1111 4587 ... 2454 3013 3144]

    def on_epoch_end(self):
        self.reset()

    def __iter__(self):
        return self


class DiscriminatorGenerator(Sequence):
    """Generate generator pretraining data.
    # Arguments
        path_pos: str, path to true data
        path_neg: str, path to generated data
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool.

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        X, Y = generator.__getitem__(idx=11)
        print(X[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(Y)
        >>> 0, 1, 1, 0, 1, 0, 0, ..., 1

        id2word = generator.id2word

        x_words = [id2word[id] for id in X[0]]
        print(x_words)
        >>> I have a <UNK> </S> <PAD> ... <PAD>
    """

    def __init__(self, path_pos, order, path_neg, B, T=40, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<S>"
        self.EOS_TOKEN = "</S>"
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.B = B
        self.T = T
        self.min_count = min_count

        sentences = load_data(path_pos, order)
        self.rows = sentences

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)

        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)

        self.n_data_pos = len(self.rows)  # Number of original data rows

        if os.path.isfile(path_neg):
            with open(path_neg, encoding="utf-8") as f:
                self.n_data_neg = sum(1 for line in f)  # Number of rows of generated data
        else:
            self.n_data_neg = 0

        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        """Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            X: numpy.array, shape = (B, max_length)
            Y: numpy.array, shape = (B, ) ,label:true=1,generated data=0.
        """
        X, Y = [], []
        start = (idx - 1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0

        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                sentence = self.rows[idx]
                words = []
                for i in sentence:
                    words.append(i)
            elif is_pos == 0:
                sentence = linecache.getline(str(self.path_neg), idx)  # str  # Read the idx row of the generated data
                words = sentence.strip().split()
            ids = sentence_to_ids(self.vocab, words)

            x = []
            x.extend(ids)
            x.append(self.EOS)  # ex. [8, 10, 6, 3, EOS]
            X.append(x)
            Y.append(is_pos)

            max_length = max(max_length, len(x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(X):
            X[i] = X[i][:max_length]  # Remove the part that exceeds the maximum length

        X = [
            pad_seq(sen, max_length) for sen in X
        ]  # The end of the current part to the maximum length part of the complement 0
        X = np.array(X, dtype=np.int32)

        return (X, Y)

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(
            start=1,
            stop=self.n_data_pos + 1,
        )  # Get an array starting from 1, with the size of the original data rows ex. [1,2,3]
        neg_indices = -1 * np.arange(
            start=1,
            stop=self.n_data_neg + 1,
        )  # Get an array starting from -1 with the size of the number of rows of generated data ex. [-1,-2,-3,-4]
        self.indicies = np.concatenate([pos_indices, neg_indices])  # Link,ex. [1,2,3,-1,-2,-3,-4]

        if self.shuffle:
            random.shuffle(self.indicies)  # disordered [-1... -500 1..n_data_pos]

    def on_epoch_end(self):
        self.reset()

    def __iter__(self):
        return self
