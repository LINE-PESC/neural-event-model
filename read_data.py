'''
Process data and prepare inputs for Neural Event Model.
'''

import bz2
import gzip
import json
import logging
import numpy as np
import sys

from gensim import models
from scipy.sparse import csr_matrix
from six import iteritems
from sklearn.preprocessing import normalize, LabelEncoder
from typing import List

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='[%(asctime)s]%(levelname)s(%(name)s): %(message)s')
LOGGER = logging.getLogger(__name__)


class DataProcessor:
    '''
    Read in data in json format, index and vectorize words, preparing data for train or test.
    '''
    def __init__(self):
        # All types of arguments seen by the processor. A0, A1, etc.
        self.arg_types = []
        self.max_sentence_length = None
        self.max_arg_length = None
        self.word_index = {"NONE": 0, "UNK": 1}    # None is padding, UNK is OOV.
        self.label_encoder = None
        self.set_labels = set()
    
    def index_data(self, filename, tokenize=None, add_new_words=True, pad_info=None, include_sentences_in_events=False, \
                   use_event_structure=True, min_args_event=1):
        '''
        Read data from file, and return indexed inputs. If this is for test, do not add new words to the
        vocabulary (treat them as unk). pad_info is applicable when we want to pad data to a pre-specified
        length (for example when testing, we want to make the sequences the same length as those from train).
        '''
        rows_buffer = []
        indexed_data = []
        open_file = gzip.open if filename.endswith('.gz') else (bz2.open if filename.endswith('.bz2') else open)
        count_rows = 0
        for row in open_file(filename, mode='rt', encoding='utf-8', errors='replace'):
            rows_buffer.append(row)
            count_rows += 1
            if (len(rows_buffer) >= 1000):
                indexed_data.extend(self._index_data_batch(rows_buffer, tokenize, add_new_words, include_sentences_in_events, min_args_event=min_args_event))
                rows_buffer.clear()
        indexed_data.extend(self._index_data_batch(rows_buffer, tokenize, add_new_words, include_sentences_in_events, min_args_event=min_args_event))
        LOGGER.info(f"INDEXED DATA/ROWS: {len(indexed_data)}/{count_rows}")
        inputs, labels = self.pad_data(indexed_data, pad_info, use_event_structure)
        return inputs, self._make_one_hot(labels)
    
    def _index_data_batch(self, rows_batch, tokenize=None, add_new_words=True, include_sentences_in_events=False, \
                          min_event_structure=1, max_event_structure=1, min_args_event=1):
        indexed_data = []
        for row in rows_batch:
            row = row.strip()
            row = row if row.startswith(('{')) else '{' + '{'.join(row.split('{')[1:])
            row = row if row.endswith(('}')) else '}'.join(row.split('}')[:-1]) + '}'
            try:
                datum = json.loads(row)
                indexed_sentence = self._index_string(datum["sentence"], tokenize=tokenize, add_new_words=add_new_words)
                datum_event_structure = datum["event_structure"]
                if isinstance(datum_event_structure, list):
                    len_event_structure = len(datum_event_structure)
                    if (len_event_structure > 0) \
                    and ((min_event_structure is None) or (len_event_structure >= max(min_event_structure, 0))) \
                    and ((max_event_structure is None) or (len_event_structure <= max(max_event_structure, 1))):                            
                        # consider only first event level 
                        datum_event_structure = datum_event_structure[0]
                    else:
                        # discards sentences without event or without number of event levels expected and continue reading
                        continue
                if (min_args_event is not None) and (len(datum_event_structure.keys()) < max(min_args_event, 1)):
                    # discards sentences with a number of insufficient arguments from an event
                    continue
                indexed_event_args = {key: self._index_string(datum_event_structure[key], tokenize=tokenize, add_new_words=add_new_words) \
                                      for key in datum_event_structure.keys()}
                if include_sentences_in_events:
                    indexed_event_args["sentence"] = indexed_sentence
                try:
                    label = datum["meta_info"][0]
                except:
                    try:
                        label = datum["meta_info"]["label"]
                    except:
                        label = None
                if label is None:
                    indexed_data.append((indexed_sentence, indexed_event_args))
                else:
                    indexed_data.append((indexed_sentence, indexed_event_args, label))
            except json.decoder.JSONDecodeError:
                if (len(row.strip()) > 0):
                    warn_msg = f"ERROR ON INDEX_DATA: The row isn't in json format: '{row}'"
                    LOGGER.warn(warn_msg)
        return indexed_data
    
    def _index_string(self, string: str, tokenize=None, add_new_words=True):
        tokens = string.lower().split() if (tokenize is None) else tokenize(string)
        for token in tokens:
            if token not in self.word_index and add_new_words:
                self.word_index[token] = len(self.word_index)
        token_indices = [self.word_index[token] if token in self.word_index else self.word_index["UNK"] \
                                         for token in tokens]
        return token_indices

    def _make_one_hot(self, labels, label_encoder=None):
        '''
        Takes labels and converts them into one hot representations.
        '''
        try:
            _ = self.label_encoder
        except AttributeError:
            self.label_encoder = None
        try:
            _ = self.set_labels
        except AttributeError:
            self.set_labels = set()
        
        if labels is None:
            return None
        if (label_encoder is not None):
            self.label_encoder = label_encoder
        else:
            if (self.label_encoder is None):
                self.label_encoder = LabelEncoder()
                try:
                    self.label_encoder.fit(self.set_labels)
                except ValueError:
                    pass
        self.label_encoder.fit(labels)
        self.set_labels.update(self.label_encoder.classes_)
        return self.label_encoder.transform(labels) 

    def pad_data(self, indexed_data, pad_info, use_event_structure=True):
        '''
        Takes a list of tuples containing indexed sentences, indexed event structures and labels, and returns numpy
        arrays.
        '''
        sentence_inputs = []
        # Setting max sentence length
        if not pad_info:
            pad_info = {}
        labels = None
        if len(indexed_data[0]) > 2:
            indexed_sentences, indexed_event_structures, labels = zip(*indexed_data)
            labels = np.asarray(labels)
        else:
            indexed_sentences, indexed_event_structures = zip(*indexed_data)
        event_structures_have_sentences = False
        if "sentence" in indexed_event_structures[0]:
            # This means index_data included sentences in event structures. We need to pad accordingly.
            event_structures_have_sentences = True
        if "max_sentence_length" in pad_info:
            self.max_sentence_length = pad_info["max_sentence_length"]
        else:
            self.max_sentence_length = max([len(indexed_sentence) for indexed_sentence in indexed_sentences])
        # Padding and/or truncating sentences
        for indexed_sentence in indexed_sentences:
            sentence_inputs.append(csr_matrix(self._pad_indexed_string(indexed_sentence, self.max_sentence_length)))

        # Removing unnecessary arguments.
        if "wanted_args" in pad_info:
            self.arg_types = list(pad_info["wanted_args"])
            if "V" not in self.arg_types:
                self.arg_types = ["V"] + self.arg_types
            if "sentence" not in self.arg_types and event_structures_have_sentences:
                self.arg_types += ["sentence"]
        else:
            arg_types = []
            for event_structure in indexed_event_structures:
                arg_types += event_structure.keys()
            self.arg_types = list(set(arg_types))
        # Making ordered event argument indices, converting argument dicts into lists with a canonical order.
        ordered_event_structures = []
        for event_structure in indexed_event_structures:
            ordered_event_structure = [event_structure[arg_type] if arg_type in event_structure else \
                                       [self.word_index["NONE"]] for arg_type in self.arg_types]
            ordered_event_structures.append(ordered_event_structure)
        if "max_arg_length" in pad_info:
            self.max_arg_length = pad_info["max_arg_length"]
        else:
            self.max_arg_length = max([max([len(arg) for arg in structure]) \
                                       for structure in ordered_event_structures])
        event_inputs = []
        for event_structure in ordered_event_structures:
            event_inputs.append(csr_matrix([self._pad_indexed_string(indexed_arg, self.max_arg_length) \
                                            for indexed_arg in event_structure]))
        indexed_sentences = None
        indexed_event_structures = None
        ordered_event_structures = None
        if use_event_structure:
            sentence_inputs = None
            inputs = np.asarray(event_inputs)
        else:
            event_inputs = None
            inputs = np.asarray(sentence_inputs)
        return inputs, labels

    def _pad_indexed_string(self, indexed_string: List[int], max_string_length: int):
        '''
        Pad and/or truncate an indexed string to the max length. Both padding and truncation happen from the left.
        '''
        string_length = len(indexed_string)
        # Padding on or truncating from the left
        padded_string = ([self.word_index["NONE"]] * (max_string_length - string_length) \
                         + indexed_string)[-max_string_length:]
        return padded_string

    def get_pad_info(self):
        '''
        Returns the information required to pad or truncate new datasets to make new inputs look like those
        processed so far. This is useful to make test data the same size as train data.
        '''
        pad_info = {}
        if self.arg_types is not None:
            pad_info["wanted_args"] = self.arg_types
        if self.max_arg_length is not None:
            pad_info["max_arg_length"] = self.max_arg_length
        if self.max_sentence_length is not None:
            pad_info["max_sentence_length"] = self.max_sentence_length
        return pad_info

    def get_embedding(self, embedding_file, add_extra_words = False):
        '''
        Reads in a pretrained embedding file, and returns a numpy array with vectors for words in word index.
        '''
        LOGGER.info("Begin of reading pretrained word embeddings ...")
        if ('.txt' in embedding_file):
            (pretrained_embedding, embedding_size) = self._get_embedding_from_txt(embedding_file)
        else:
            (pretrained_embedding, embedding_size) = self._get_embedding_from_bin(embedding_file)
        if add_extra_words:
            # adding words pretrained still aren't in word_index        
            tokens = list(pretrained_embedding.keys() - self.word_index.keys())
            for token in tokens:
                self.word_index[token] = len(self.word_index)
        embedding = np.array(list(pretrained_embedding.values()))
        low_embedding = embedding.min(axis=0)
        high_embedding = embedding.max(axis=0) + np.finfo(embedding.dtype).eps
        len_word_index = len(self.word_index)
        shape_embedding = (len_word_index, embedding_size)
        embedding = np.random.uniform(low_embedding, high_embedding, shape_embedding)
        count_words_pretrained_embedding = 0
        for word in self.word_index:
            if word in pretrained_embedding:
                embedding[self.word_index[word]] = pretrained_embedding[word]
                count_words_pretrained_embedding += 1
        # normalize embedding features with l2-norm
        embedding = normalize(embedding, axis=0)
        embedding[self.word_index["NONE"]] = np.zeros(embedding_size)
        #embedding[self.word_index["UNK"]] = np.zeros(embedding_size)
        LOGGER.info("End of reading pretrained word embeddings.")
        proportion = (count_words_pretrained_embedding * 100.0) / len_word_index
        string_proportion = f"Proportion of pre-embedding words: {proportion:.2f}% ({count_words_pretrained_embedding} / {len_word_index})."
        if add_extra_words:
            string_proportion = f"{string_proportion}\tIncluding {len(tokens)} extra tokens."
        string_sep = "=" * len(string_proportion)
        LOGGER.info(string_sep)
        LOGGER.info(string_proportion)
        LOGGER.info(string_sep)
        return embedding, count_words_pretrained_embedding
    
    def _get_embedding_from_bin(self, embedding_file):
        '''
        Reads in a pretrained embedding bin file, and returns a numpy array with vectors for words in word index.
        '''
        model = models.keyedvectors.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        pretrained_embedding = {}
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item:-item[1].count):
            pretrained_embedding[word] = np.asarray(model.syn0[vocab.index])
        embedding_size = model.syn0.shape[1]
        return (pretrained_embedding, embedding_size)

    def _get_embedding_from_txt(self, embedding_file):
        '''
        Reads in a pretrained embedding txt file, and returns a numpy array with vectors for words in word index.
        '''
        pretrained_embedding = {}
        open_file = gzip.open if embedding_file.endswith('.gz') else (bz2.open if embedding_file.endswith('.bz2') else open)
        for line in open_file(embedding_file, mode='rt', encoding='utf-8'):
            parts = line.strip().split()
            if len(parts) == 2:
                continue
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            pretrained_embedding[word] = np.asarray(vector)
        embedding_size = len(vector)
        return (pretrained_embedding, embedding_size)

    def get_vocabulary_size(self):
        '''
        Returns the number of unique words seen in indexed data.
        '''
        return len(self.word_index)
