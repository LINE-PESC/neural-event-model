'''
Process data and prepare inputs for Neural Event Model.
'''

import sys
import logging

import os
import bz2
import gzip
import json
import math
import numpy as np
import time
import asyncio

from datetime import datetime
from gensim import models
from scipy.sparse import csr_matrix
from six import iteritems
from sklearn.preprocessing import normalize, LabelEncoder
from typing import List
from builtins import isinstance
from contextlib import contextmanager, asynccontextmanager
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='[%(asctime)s]%(levelname)s(%(name)s): %(message)s')
LOGGER = logging.getLogger(__name__)

BUFFER_HINT = 2 ** 29 # 512MiB

class DataProcessor:
    '''
    Read in data in json format, index and vectorize words, preparing data for train or test.
    '''

    def __init__(self):
        # All types of arguments seen by the processor. A0, A1, etc.
        self.arg_types = []
        self.max_sentence_length = None
        self.max_arg_length = None
        self.word_index = {"NONE": 0, "UNK": 1}  # NONE is padding, UNK is OOV.
        self.label_encoder = None
        self.set_labels = set()


    def index_data(self, filename, tokenize=None, add_new_words=True,
                   pad_info=None, include_sentences_in_events=False,
                   use_event_structure=True, min_event_structure=1,
                   max_event_structure=1, min_args_event=1,
                   return_data=False, verbose=False):
        result = asyncio.run(self._async_index_data(
                            filename, tokenize, add_new_words,
                            pad_info, include_sentences_in_events,
                            use_event_structure, min_event_structure,
                            max_event_structure, min_args_event,
                            return_data, verbose))
        return result


    async def _async_index_data(self, filename, tokenize=None, add_new_words=True,
                   pad_info=None, include_sentences_in_events=False,
                   use_event_structure=True, min_event_structure=1,
                   max_event_structure=1, min_args_event=1,
                   return_data=False, verbose=False):
        '''
        Read data from file, and return indexed inputs. If this is for test, do not add new words to the
        vocabulary (treat them as unk). pad_info is applicable when we want to pad data to a pre-specified
        length (for example when testing, we want to make the sequences the same length as those from train).
        '''
        rows_buffer = []
        indexed_data = []
        count_rows = 0
        BUFFER_ROWS = 1000
        async with async_read_txt_file(filename, buffer_hint=BUFFER_HINT,
                                       encoding='utf-8', errors='replace',
                                       verbose=verbose) as opened_file:
            async for row in opened_file:
                rows_buffer.append(row)
                count_rows += 1
                if (len(rows_buffer) >= BUFFER_ROWS):
                    indexed_data.extend(self._index_data_batch(rows_buffer,
                                                               tokenize,
                                                               add_new_words,
                                                               include_sentences_in_events,
                                                               min_event_structure=min_event_structure,
                                                               max_event_structure=max_event_structure,
                                                               min_args_event=min_args_event,
                                                               return_data=return_data))
                    rows_buffer.clear()
        # end-for
        if rows_buffer:
            indexed_data.extend(self._index_data_batch(rows_buffer, tokenize, add_new_words,
                                                       include_sentences_in_events,
                                                       min_event_structure=min_event_structure,
                                                       max_event_structure=max_event_structure,
                                                       min_args_event=min_args_event,
                                                       return_data=return_data))
        LOGGER.info(f"INDEXED DATA/ROWS: {len(indexed_data)}/{count_rows} (with min of {min_args_event} args)")
        inputs, labels, datasrc = self.pad_data(indexed_data, pad_info, use_event_structure, return_data=return_data)
        return (inputs, self._make_one_hot(labels), datasrc) if return_data else (inputs, self._make_one_hot(labels))


    def _index_data_batch(self, rows_batch, tokenize=None, add_new_words=True, include_sentences_in_events=False, \
                          min_event_structure=1, max_event_structure=1, min_args_event=1, return_data=False):
        indexed_data = []
        for row in rows_batch:
            row = row.strip()
            row = row if row.startswith(('{')) else '{' + '{'.join(row.split('{')[1:])
            row = row if row.endswith(('}')) else '}'.join(row.split('}')[:-1]) + '}'
            datum = None
            try:
                datum = json.loads(row)
            except json.decoder.JSONDecodeError:
                if (len(row.strip()) > 0):
                    warn_msg = f"ERROR ON INDEX_DATA: The row isn't in json format: '{row}'"
                    LOGGER.warning(warn_msg)
                continue
            indexed_sentence = self._index_string(datum["sentence"], tokenize=tokenize, add_new_words=add_new_words)
            datum_event_structure = datum["event_structure"]
            list_datum_event_structure = []
            if isinstance(datum_event_structure, list):
                len_event_structure = len(datum_event_structure)
                if (len_event_structure > 0) \
                and ((min_event_structure is None) or (len_event_structure >= max(min_event_structure, 0))) \
                and ((max_event_structure is None) or (len_event_structure <= max(max_event_structure, 1))):                            
                    list_datum_event_structure = datum_event_structure #= datum_event_structure[0]
                else:
                    # discards sentences without event or without number of event levels expected and continue reading
                    continue
            else:
                list_datum_event_structure = [datum_event_structure]
            for datum_event_structure in list_datum_event_structure:
                if (min_args_event is not None) and (len(datum_event_structure.keys()) < max(min_args_event, 1)):
                    # discards sentences with a number of insufficient arguments from an event
                    continue
                indexed_event_args = {key: self._index_string(datum_event_structure[key], tokenize=tokenize, add_new_words=add_new_words) \
                                      for key in datum_event_structure.keys()}
                
                # After index with stemming some args could be empty, so filter again
                indexed_event_args = {key: value for key, value in indexed_event_args.items() if len(value) > 0}
                if (min_args_event is not None) and (len(indexed_event_args.keys()) < max(min_args_event, 1)):
                    # discards sentences with a number of insufficient arguments from an event
                    continue
                
                if include_sentences_in_events:
                    indexed_event_args["sentence"] = indexed_sentence
                indexed_row = [indexed_sentence, indexed_event_args]
                try:
                    label = datum["meta_info"][0]
                    indexed_row.append(label) # for test phase
                except:
                    try:
                        label = datum["label"]
                        indexed_row.append(label) # for test phase
                    except:
                        pass # only for training phase
                if return_data:
                    indexed_row.append(datum)
                indexed_data.append(tuple(indexed_row))
        return indexed_data
    
    def _index_string(self, string: str, tokenize=None, add_new_words=True):
        tokens = self.apply_tokenize_func(string, tokenize).lower().split()
        for token in tokens:
            if token not in self.word_index and add_new_words:
                self.word_index[token] = len(self.word_index)
        token_indices = [self.word_index[token] if token in self.word_index else self.word_index["UNK"] \
                                         for token in tokens]
        return token_indices

    def apply_tokenize_func(self, string: str, tokenize=None):
        tokenize = [] if (tokenize is None) else (list(tokenize) if isinstance(tokenize, (list, tuple)) else [tokenize])
        for tokenizer in tokenize:
            tokens = tokenizer(string)
            string = " ".join(tokens)
        return string

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

    def pad_data(self, indexed_data, pad_info, use_event_structure=True, return_data=False):
        '''
        Takes a list of tuples containing indexed sentences, indexed event structures and labels, and returns numpy
        arrays.
        '''
        # Setting max sentence length
        if not pad_info:
            pad_info = {}
        labels = None
        datasrc = None
        len_indexed_data = len(indexed_data[0])
        zip_indexed_data = zip(*indexed_data)
        if len_indexed_data > 3:
            indexed_sentences, indexed_event_structures, labels, datasrc = zip_indexed_data
            labels = np.asarray(labels)
            datasrc = np.asarray(datasrc)
        elif len_indexed_data == 3:
            if return_data:
                indexed_sentences, indexed_event_structures, datasrc = zip_indexed_data
                datasrc = np.asarray(datasrc)
            else:
                indexed_sentences, indexed_event_structures, labels = zip_indexed_data
                labels = np.asarray(labels)
        else:
            indexed_sentences, indexed_event_structures = zip_indexed_data
        event_structures_have_sentences = False
        if "sentence" in indexed_event_structures[0]:
            # This means index_data included sentences in event structures. We need to pad accordingly.
            event_structures_have_sentences = True
        if "max_sentence_length" in pad_info:
            self.max_sentence_length = pad_info["max_sentence_length"]
        else:
            self.max_sentence_length = max([len(indexed_sentence) for indexed_sentence in indexed_sentences])
        # Padding and/or truncating sentences

        sentence_inputs = []
        event_inputs = []
        if not use_event_structure:
            for indexed_sentence in indexed_sentences:
                sentence_inputs.append(csr_matrix(self._pad_indexed_string(indexed_sentence, self.max_sentence_length)))
        else:
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
                self.arg_types = sorted(set(arg_types))
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
        
        return inputs, labels, datasrc

    def _pad_indexed_string(self, indexed_string: List[int], max_string_length: int):
        '''
        Pad and/or truncate an indexed string to the max length. Both padding and truncation happen from the left.
        '''
        max_string_length = int(pow(2, int(math.log(max_string_length, 2)) + 1))
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
        if self.arg_types is not None and len(self.arg_types) > 0:
            pad_info["wanted_args"] = self.arg_types
        if self.max_arg_length is not None:
            pad_info["max_arg_length"] = self.max_arg_length
        if self.max_sentence_length is not None:
            pad_info["max_sentence_length"] = self.max_sentence_length
        return pad_info

    def get_embedding(self, embedding_file, add_extra_words=False, verbose=False):
        '''
        Reads in a pretrained embedding file, and returns a numpy array with vectors for words in word index.
        '''
        LOGGER.info("Begin of reading pretrained word embeddings ...")
        if ('.txt' in embedding_file):
            (pretrained_embedding, embedding_dim) = self._get_embedding_from_txt(embedding_file, verbose)
        else:
            (pretrained_embedding, embedding_dim) = self._get_embedding_from_bin(embedding_file)
        if add_extra_words:
            # adding words pretrained still aren't in word_index        
            tokens = list(pretrained_embedding.keys() - self.word_index.keys())
            for token in tokens:
                self.word_index[token] = len(self.word_index)
        len_word_index = len(self.word_index)
        shape_embedding = (len_word_index, embedding_dim)
        #embedding_matrix = np.array(list(pretrained_embedding.values()))
        # eps = np.finfo(embedding_matrix.dtype).eps
        # low_embedding = embedding_matrix.min(axis=0)
        # high_embedding = embedding_matrix.max(axis=0) + eps
        # LOGGER.info(f"EMBEDDING LOW: {low_embedding.min()}\tEMBEDDING HIGH: {high_embedding.min()}\tEMBEDDING MIN-ABS: {np.amin(np.absolute(embedding_matrix))}")
        embedding_matrix = np.zeros(shape_embedding)  # np.random.uniform(low_embedding, high_embedding, shape_embedding)
        count_words_pretrained_embedding = 0
        for word in self.word_index:
            if word in pretrained_embedding:
                embedding_matrix[self.word_index[word]] = pretrained_embedding[word]
                count_words_pretrained_embedding += 1
        low_embedding = embedding_matrix.min(axis=0)
        high_embedding = embedding_matrix.max(axis=0)
        LOGGER.info(f"EMBEDDING LOW: {low_embedding.min()}\tEMBEDDING HIGH: {high_embedding.min()}")   
        # Each term without word-embedding receives a representation very close to the origin of the vector space, but not zero.
        embedding_matrix[self.word_index["UNK"]] += np.finfo(embedding_matrix.dtype).eps     
        # normalize embeddings with l2-norm
        # axis used to normalize the data along. If 1, independently normalize each sample, otherwise (if 0) normalize each feature
        embedding_matrix = normalize(embedding_matrix, axis=1)
        # embedding[self.word_index["NONE"]] = np.zeros(embedding_dim)
        low_embedding = embedding_matrix.min(axis=0)
        high_embedding = embedding_matrix.max(axis=0)

        LOGGER.info(f"NORMALIZED EMBEDDING LOW: {low_embedding.min()}\tNORMALIZED EMBEDDING HIGH: {high_embedding.min()}")
        LOGGER.info(f"Word embedding shape: {embedding_matrix.shape}")
        LOGGER.info("End of reading pretrained word embeddings.")
        
        proportion = (count_words_pretrained_embedding * 100.0) / len_word_index
        string_proportion = f"Proportion of pre-embedding words: {proportion:.2f}% ({count_words_pretrained_embedding} / {len_word_index})."
        if add_extra_words:
            string_proportion = f"{string_proportion}\tIncluding {len(tokens)} extra tokens."
        string_sep = "=" * len(string_proportion)
        LOGGER.info(string_sep)
        LOGGER.info(string_proportion)
        LOGGER.info(string_sep)
        return embedding_matrix, count_words_pretrained_embedding
    
    def _get_embedding_from_bin(self, embedding_file):
        '''
        Reads in a pretrained embedding bin file, and returns a numpy array with vectors for words in word index.
        '''
        model = models.keyedvectors.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        pretrained_embedding = {}
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item:-item[1].count):
            pretrained_embedding[word] = np.asarray(model.syn0[vocab.index])
        embedding_dim = model.syn0.shape[1]
        return (pretrained_embedding, embedding_dim)

    def _get_embedding_from_txt(self, embedding_file, verbose=False):
        result = asyncio.run(self._async_get_embedding_from_txt(embedding_file, verbose))
        return result

    
    async def _async_get_embedding_from_txt(self, embedding_file, verbose=False):
        '''
        Reads in a pretrained embedding txt file, and returns a numpy array with vectors for words in word index.
        '''
        array_coefs = None
        embedding_dim = None
        pretrained_embedding = {}

        async with async_read_txt_file(embedding_file,
                                       buffer_hint=BUFFER_HINT,
                                       encoding='utf-8',
                                       errors='replace',
                                       verbose=verbose) as opened_file:
            LOGGER.info(f"Reading pretrained word embeddings from file: {embedding_file}")
            async for line in opened_file:
                line = line.strip()
                if line and (' ' in line):
                    word, coefs = line.split(maxsplit=1)
                    array_coefs = np.fromstring(coefs, "f", sep=" ")
                    if len(array_coefs) > 1:
                        pretrained_embedding[word] = array_coefs
        embedding_dim = pretrained_embedding[list(pretrained_embedding.keys())[0]].shape[0]
        return (pretrained_embedding, embedding_dim)

    def get_vocabulary_size(self):
        '''
        Returns the number of unique words seen in indexed data.
        '''
        return len(self.word_index)
    

@contextmanager
def file_txt_buffered(filename: str,
                      buffer_hint: int = -1,
                      encoding='utf-8',
                      errors=None,
                      verbose=False):
    log_level = LOGGER.getEffectiveLevel()
    if verbose:
        LOGGER.setLevel(logging.DEBUG)
    # TODO refatorar todas as chamadas para ler arquivos compactados para desconderar a primeira linha se estiver nessa condição: line.startswith("/") and line.endswith("''")
    open_file = (gzip.open if filename.endswith('.gz') \
                    else (bz2.open if filename.endswith('.bz2') \
                        else open))
    multiply_buffer = 3 if filename.endswith('.bz2') else 1
    buffer_hint = max(buffer_hint, BUFFER_HINT)
    buffer_hint = min(buffer_hint, os.path.getsize(filename) * multiply_buffer)
    
    kwargs = {'mode': 'rt'}
    if encoding is not None:
        kwargs.update({'encoding': encoding})
    if errors is not None:
        kwargs.update({'errors': errors})
    with open_file(filename, **kwargs) as opened_file:
        def _readlines_():
            LOGGER.debug(f"Reading lines from file {filename}")
            lines = opened_file.readlines(buffer_hint)
            if lines:
                LOGGER.debug(f"{len(lines)} lines read from file {filename}")
            else:
                LOGGER.debug(f"End of file reading: {filename}")
            return lines
        def _gen_():
            lines = _readlines_()
            while lines:
                for line in lines:
                    yield line
                lines = _readlines_()
        yield _gen_()
    LOGGER.setLevel(log_level)

@asynccontextmanager
async def async_read_txt_file(filename: str,
                              buffer_hint: int = -1,
                              encoding='utf-8',
                              errors=None,
                              verbose=False):
    log_level = LOGGER.getEffectiveLevel()
    if verbose:
        LOGGER.setLevel(logging.DEBUG)
    # TODO refatorar todas as chamadas para ler arquivos compactados para desconderar a primeira linha se estiver nessa condição: line.startswith("/") and line.endswith("''")
    open_file = (gzip.open if filename.endswith('.gz') \
                    else (bz2.open if filename.endswith('.bz2') \
                        else open))
    multiply_buffer = 3 if filename.endswith('.bz2') else 1
    buffer_hint = max(buffer_hint, BUFFER_HINT)
    buffer_hint = min(buffer_hint, os.path.getsize(filename) * multiply_buffer)

    kwargs = {'mode': 'rt'}
    if encoding is not None:
        kwargs.update({'encoding': encoding})
    if errors is not None:
        kwargs.update({'errors': errors})
    LOGGER.info(f"Opening file {filename} with buffer hint {buffer_hint} and keyword arguments {kwargs}...")
    
    with open_file(filename, **kwargs) as opened_file:
        def _readlines_(times_read=0):
            LOGGER.debug(f"(#{times_read}) Reading lines from file {filename}")
            # may be slow as it has disk access
            lines = opened_file.readlines(buffer_hint)
            if lines:
                LOGGER.debug(f"(#{times_read}) {len(lines)} lines read from file {filename}")
            else:
                LOGGER.debug(f"(#{times_read}) End of file reading: {filename}")
            return lines
        async def _gen_():
            times_read = 0
            start = datetime.now()
            lines = _readlines_()
            diff_seconds = (datetime.now() - start).total_seconds()
            while lines:
                times_read += 1
                LOGGER.debug(f"(#{times_read}) Preparing to assynchronously read more lines from file {filename}")
                thread_io = asyncio.to_thread(_readlines_, times_read)
                tasks = asyncio.gather(thread_io)
                await asyncio.sleep(diff_seconds * .1) # to allow the event loop to take control
                LOGGER.debug(f"(#{times_read - 1}) Yielding {len(lines)} lines from file {filename}")
                for line in lines:
                    yield line
                LOGGER.debug(f"(#{times_read}) Waiting for next lines from file {filename}")
                lines = await tasks
                lines = lines[0]
                del thread_io
                del tasks
        yield _gen_()
    LOGGER.setLevel(log_level)
