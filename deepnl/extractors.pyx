# -*- coding: utf-8 -*-
# distutils: language = c++
# cython: profile=True

"""
Feature extractors.

Each extractor maintains its own table, for mapping features to vectors.
Some of them also maintain a dictionary, for mapping tokens to features.
They are resposible for loading and saving these data to/from a model file.
"""

# standard
import logging
import numpy as np
import os
import re
from collections import Counter, OrderedDict
import cPickle as pickle
from itertools import izip
from numpy import int32 as INT
import sys                      # modules
import itertools

# local
from word_dictionary import WordDictionary as WD
import embeddings
from utils import Trie, strip_accents

# ----------------------------------------------------------------------

cdef class Iterable:
    """
    ABC for classes that provide the __iter__() method.
    """

    def __iter__(self):
        return self

# ----------------------------------------------------------------------

cdef class ConvertGenerator(Iterable):
    cdef Converter converter
    cdef sentences
    cdef converted
    cdef bool cache

    def __init__(self, converter, sentences, cache=False):
        """
        :param sentences: an iterable over sentences.
        :param cache: if this is True, caches converted sentences,
           avoiding repeated conversion. Useful if sentences are few enough to
           fit in memory.
        """
        self.converter = converter
        self.sentences = sentences
        self.converted = []     # cache converted sentences
        self.cache = cache

    def __iter__(self):
        cdef np.ndarray c
        if self.converted:
            for s in self.converted:
                yield s
        else:
            for s in self.sentences:
                c =  self.converter.convert(s)
                if self.cache:
                    self.converted.append(c)
                yield c

cdef class Converter(object):
    """
    Interface to the extractors.
    Extracts features from a sentence and converts them into a list of feature
    vectors in feature space.
    """
    
    # one extractor for each feature
    # cpdef list extractors

    def __init__(self):
        self.extractors = []
        self.fields = []
    
    cpdef int_t size(self):
        #return sum(e.size() for e in self.extractors)
        cdef int_t s = 0
        for e in self.extractors:
            s += e.size()
        return s

    def add(self, extractor, field=-1):
        """
        Adds an extractor function to this Converter.
        In order to get a token's feature indices, the Converter will call
        each of its extraction functions passing the token as a parameter.

        :param extractor: the extractor to add.
        :param field: which token field to pass to the extractor. The whole token if -1.

        # If an extractor needs all token fields, it must redefine method extract()
        """
        self.extractors.append(extractor)
        self.fields.append(field)
    
    def generator(self, sentences, cache=False):
        """
        :return: a generator for converting the :param sentences:.
        """
        return ConvertGenerator(self, sentences, cache)

    cdef np.ndarray[int_t] get_padding_left(self):
        """
        :return: the features of the left padding token.
        
        """
        return INT([e.get_padding_left() for e in self.extractors])
    
    cdef np.ndarray[int_t] get_padding_right(self):
        """
        :return: the features of the right padding token.
        """
        return INT([e.get_padding_right() for e in self.extractors])
    
    cpdef np.ndarray[int_t,ndim=2] convert(self, list sent, dict other=None):
        """
        Converts a sentence into an array of feature indices.
        :param sent: a list of tokens.
	:param other: a dictionary with other params.
        :return: an array of all extractors' results.
        """
        return INT(zip(*[(<Extractor>e).extract(sent, field, other) for e, field in zip(self.extractors, self.fields)]))
        # CHECKME: is this faster?
        # return np.array([extractor.extract(sent) for extractor in self.extractors]).T

    cpdef np.ndarray[float_t] lookup(self,
                                     np.ndarray[int_t,ndim=2] sentence,
                                     np.ndarray out=None):
        """
        Collect the feature vectors of all sentence tokens.
        :param sentence: Each row represents a token through its indices into
            the corresponding feature table.
        :param out: vector where to store result.
        :return: a single feature vector, combining the vectors of all features
            of eack token in :param sentence:
        """
        if out is None:
            out = np.empty(self.size() * len(sentence))
        cdef int_t start = 0, end
        for token in sentence:
            for feature, extractor in izip(token, self.extractors):
                end = start + extractor.size()
                np.copyto(out[start:end], extractor[feature])
                start = end
        return out

    cpdef adaGradInit(self, float_t adaEps):
        """
        Initialize AdaGrad.
        """
        for e in self.extractors:
            e.adaGradInit(adaEps)

    cpdef update(self, np.ndarray[float_t] grads, np.ndarray[int_t,ndim=2] sequence,
                 float_t learning_rate):
        """
        Update the features according to the given gradients.
        :param grads: vector of feature gradients.
        :param larning_rate: learning rate multiplier.
        :param sequence: each row represents a token through its indices into
            each feature table. Includes padding.
        """
        cdef int_t start = 0, end
        for token in sequence:
            for feature, extractor in izip(token, self.extractors):
                end = start + extractor.size()
                if extractor.adaGrads is None:
                    extractor.table[feature] += learning_rate * grads[start:end]
                else:
                    # AdaGrad
                    extractor.adaGrads[feature] += grads[start:end] * grads[start:end]
                    extractor.table[feature] += learning_rate * grads[start:end] / np.sqrt(extractor.adaGrads[feature])
                start = end

    def save(self, file):
        """
        Save all extractors' data to file.
        """
        # save fields
        pickle.dump(self.fields, file)
        # save class names
        class_names = [type(e).__name__ for e in self.extractors]
        pickle.dump(class_names, file)
        for extractor in self.extractors:
            extractor.save(file)

    def load(self, file):
        """
        Load all extractors' data from file.
        """
        # load fields
        self.fields = pickle.load(file)
        class_names = pickle.load(file)
        # FIXME: this will recognize only classes defined in this file
        #m = __import__("extractors")
        m = sys.modules['deepnl.extractors']
        for c,f in izip(class_names, self.fields):
            cls = getattr(m, c)
            self.add(cls.__new__(cls), f)
        for extractor in self.extractors:
            extractor.load(file)

# ----------------------------------------------------------------------

cdef class Extractor(object):
    """
    Abstract feature extractor.

    Each extractor deals with one kind of features, e.g. embeddings,
    pos, caps, etc.
    Each one is responsible for saving and loading its own data to/from a
    common model file.
    """

    # cdef readonly dict dict
    # cdef readonly np.ndarray table
    # cdef readonly np.ndarray adaGrads # volatile

    def __init__(self):
        self.adaGrads = None

    def __getitem__(self, feature):
        """
        Get the vector corresponding to the :param feature:
        """
        return self.table[feature]

    def __setitem__(self, feature, value):
        """
        Set the vector corresponding to the :param feature:
        """
        self.table[feature] = value

    cpdef int_t get_padding_left(self):
        ":return: the feature representing the token used as left padding"
        return <int_t>self.dict.padding_left

    cpdef int_t get_padding_right(self):
        ":return: the feature representing the token used as right padding"
        return <int_t>self.dict.padding_right

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.

        Override this if the extractor needs to deal differently with tokens.
        """
        if field >= 0:
            return [self.dict[token[field]] for token in tokens]
        else:
            return [self.dict[token] for token in tokens]

    cpdef int_t size(self):
        """
        :return: dimension of embeddings space
        """
        return self.table.shape[1]

    cpdef adaGradInit(self, float_t adaEps):
        self.adaGrads = np.full_like(self.table, adaEps);

    def save(self, file):
        pickle.dump(self.dict, file)
        pickle.dump(self.table, file)

    def load(self, file):
        self.dict = pickle.load(file)
        self.table = pickle.load(file)

# ----------------------------------------------------------------------

cdef class Embeddings(Extractor):
    """
    Lookup layer.
    """

    def __init__(self, size=0, vocab_file=None, vectors=None, vocab=None,
                 variant=None):
        """
        Construct from either precomputed vocabulary and vectors files,
        or from a list of words :param vocab:.
        :param size: vector space dimension.
        :param vocab_file: file containing the vocabulary
        :param vocab: list of vocabulary words
        :param vectors: file containing the vectors
        :param variant: style of embeddings (senna, polyglot, word2vect)
        """
        super(Embeddings, self).__init__()

        if vocab:
            self.dict = <dict>WD(None, wordlist=vocab, variant=variant)
            if vectors and os.path.exists(vectors):
                self.table = self.load_vectors(vectors)
            else:
                self.table = embeddings.generate_vectors(len(self.dict), size)
        elif variant == 'word2vec':
            # load both vocab and vectors from single file
            self.table, wordlist = embeddings.Word2Vec.load(vectors)
            self.dict = <dict>WD(None, wordlist=wordlist, variant=variant)
            # add vectors for special symbols
            extra = len(self.dict) - len(self.table)
            if extra > 0:
                logging.info('Adding %d special symbols' % extra)
                self.table = np.concatenate((self.table, embeddings.generate_vectors(extra, self.table.shape[1])))
        elif vocab_file:
            self.dict = <dict>WD(None, wordlist=self.load_vocabulary(vocab_file), variant=variant)
            if vectors and os.path.exists(vectors):
                self.table = self.load_vectors(vectors)
            else:
                self.table = embeddings.generate_vectors(len(self.dict), size)

    def merge(self, list vocab):
        """Extend the dictionary with words from list :param vocab:"""
        for word in vocab:
            self.dict.add(word) # add if not present
        # generate vectors for added words
        extra = len(self.dict) - len(self.table)
        if extra > 0:
            logging.info('Added %d words' % extra)
            self.table = np.concatenate((self.table, embeddings.generate_vectors(extra, self.table.shape[1])))

    def save(self, file):
        self.dict.save(file)
        pickle.dump(self.table, file)

    def load(self, file):
        self.dict = <dict>WD.load(file)
        self.table = pickle.load(file)

    def load_vocabulary(self, file, variant=None):
        # FIXME: allow chosing variant
        return embeddings.Plain.read_vocabulary(file)

    def save_vocabulary(self, filename):
        embeddings.Plain.write_vocabulary(self.dict.words, filename)

    def load_vectors(self, file, variant=None):
        # FIXME: allow choosing variant
        return embeddings.Plain.read_vectors(file)

    def save_vectors(self, filename, variant=None):
        if variant == 'word2vec':
            embeddings.Word2Vec.save(filename, self.dict.words, self.table)
        else:
            embeddings.Plain.write_vectors(filename, self.table)

    def lookup_ngram(self, ngramIDs):
        """
        Lookup an ngrams.
        """
        # lookup ngram IDs to obtain back words
        tokens = self.dict.get_words(ngramIDs)
        return self.dict[' '.join(tokens)]

    def sentence(self, feats):
        """
        Get sentence back from features.
        Only for debugging.
        """
        # lookup ngram IDs to obtain back words
        tokens = self.dict.get_words([tok[0] for tok in feats])
        return ' '.join(tokens)

# ----------------------------------------------------------------------
# Capitalization

cdef class Caps(object):                     # Caps(Enumeration)
    """
    Enumeration for capitalization types.
    """
    # lower = 0
    # title = 1
    # non_alpha = 2
    # other = 3
    # upper = 4                   # Attardi
    # num_values = 5

    # SENNA
    padding = 0
    upper  = 1
    hascap = 2
    title  = 3
    nocaps = 4
    num_values = 5              # extractor values

    @staticmethod
    def code(word):
        """
        Returns a code describing the capitalization of the word:
        lower, title, upper, other or non-alpha (numbers and other tokens that can't be
        capitalized).
        """
        # if not word.isalpha():
        #     return Caps.non_alpha

        # if word.istitle():
        #     return Caps.title

        # if word.islower():
        #     return Caps.lower

        # if word.isupper():          # Attardi
        #     return Caps.upper

        # return Caps.other

        if word.isupper():
            return Caps.upper

        if word[0].isupper():       # istitle() checks other letters too
            return Caps.title

        # can't use islower() because it accepts '3b'
        for c in word:
            if c.isupper():
                return Caps.hascap

        return Caps.nocaps

cdef class CapsExtractor(Extractor):

    def __init__(self, size):
        """
        :param size: dimension of vectors.
        """
        super(CapsExtractor, self).__init__()
        self.table = embeddings.generate_vectors(Caps.num_values, size)

    cpdef int_t get_padding_left(self):
        ":return: the feature representing the token used as left padding"
        return Caps.padding

    cpdef int_t get_padding_right(self):
        ":return: the feature representing the token used as right padding"
        return Caps.padding

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if -1.
        :return: the list of capitalization codes for the given list of tokens.
        """
        if field >= 0:
            return [Caps.code(tok[field]) for tok in tokens]
        else:
            return [Caps.code(w) for w in tokens]

    def save(self, file):
        # no dictionary
        pickle.dump(self.table, file)

    def load(self, file):
        self.table = pickle.load(file)

def capitalize(word, capitalization):
    """
    Capitalizes the word in the desired format. If the capitalization is 
    Caps.other, it is set all uppercase.
    """
    if capitalization == Caps.non_alpha:
        return word
    elif capitalization == Caps.lower:
        return word.lower()
    elif capitalization == Caps.title:
        return word.title()
    elif capitalization == Caps.other:
        return word.upper()
    else:
        raise ValueError("Unknown capitalization type.")

# ----------------------------------------------------------------------
    
cdef class AffixExtractor(Extractor):
    """Abstract class for prefix or suffix extractors."""

    padding = 0
    other = 1                   # NOSUFFIX
    specials = 2                # number of specials (other, padding)

    def __init__(self, size, filename=None, wordlist=[], lowcase=True):
        """
        :param size: the dimension of the embeddings space
        :param filename: load affixes from this file, if given.
        :param wordlist: extract affixes from these words, if given.
        :param lowcase: set the affix in lowercase
        """
        super(AffixExtractor, self).__init__()
        self.lowcase = lowcase
        specials = AffixExtractor.specials
        if filename:
            self.load_affixes(filename)
        else:
            affixes = self.build(wordlist, lowcase=lowcase)
            # leave reserved values for specials
            self.dict = { x: i+specials for i,x in enumerate(affixes) }
        # create vectors for possible values
        self.table = embeddings.generate_vectors(len(self.dict)+specials, size)

    cpdef int_t get_padding_left(self):
        ":return: the feature representing the token used as left padding"
        return AffixExtractor.padding

    cpdef int_t get_padding_right(self):
        ":return: the feature representing the token used as right padding"
        return AffixExtractor.padding

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :return: the list of prefix/suffix codes for the given :param tokens:.
        """
        if field >= 0:
            return [self.affix(tok[field]) for tok in tokens]
        else:
            return [self.affix(w) for w in tokens]

    def load_affixes(self, filename):
        """
        Load prefixes or suffixes from file :param filename:.
        """
        logger = logging.getLogger("Logger")
        specials = AffixExtractor.specials
        self.dict = {}
        # leave reserved values for specials
        cdef int_t i = specials
        try:
            with open(filename, 'rb') as f:
                for line in f:
                    affix = unicode(line.strip(), 'utf-8')
                    self.dict[affix] = i
                    i += 1
        except IOError:
            logger.error("File %s doesn't exist." % filename)
            raise

    def write(self, filename):
        """
        Write prefixes or suffixes to file :param filename:.
        """
        with open(filename, 'wb') as f:
            # order by ID
            affixes = [''] * len(self.dict)
            for a, i in self.dict.iteritems():
                affixes[i - self.specials] = a
            for i in range(self.specials, len(self.dict)):
                print >> f, affixes[i].encode('utf-8')

# ----------------------------------------------------------------------

cdef class SuffixExtractor(AffixExtractor):

    # max suffix length (mimic SENNA)
    max_length = 2

    def affix(self, word):
        """
        Return the suffix code for the given word.
        """
        # as in SENNA, we use the whole word in this case:
        # if len(word) <= SuffixExtractor.max_length:
        #     return Affix.other

        suffix = word[-SuffixExtractor.max_length:]
        if self.lowcase:
            suffix = suffix.lower()
        return self.dict.get(suffix, AffixExtractor.other)

    def build(self, wordlist, num=200, min_occurrences=3,
              length=SuffixExtractor.max_length, lowcase=True):
        """
        Creates a list with the most common suffixes found in wordlist.
        
        :param wordlist: a list of words (there shouldn't be repetitions)
        :param num: maximum number of suffixes
        :param min_occurrences: minimum number of occurrences of each suffix
        in wordlist
        :param length: desired length of suffixes
        :param lowcase: set the affix in lowercase
        """
        # we take the whole word if len(x) <= length
        lowcaser = lambda x: x.lower() if lowcase else x
        all_endings = (lowcaser(x[-length:]) for x in wordlist
                       if not re.search('_|\d', x[-length:]))
        c = Counter(all_endings)
        common_endings = c.most_common(num)
        return [e for e, n in common_endings if n >= min_occurrences]

# ----------------------------------------------------------------------

cdef class PrefixExtractor(AffixExtractor):

    # max prefix length (mimic SENNA)
    max_length = 2

    def affix(self, word):
        """
        Return the prefix code for the given :param word:.
        """

        # as in SENNA, we use the whole word in this case:
        # if len(word) <= PrefixExtractor.max_length:
        #     return Affix.other

        prefix = word[:PrefixExtractor.max_length]
        if self.lowcase:
            prefix = prefix.lower()
        return self.dict.get(prefix, AffixExtractor.other)

    def build(cls, wordlist, num=200, min_occurrences=3,
              length=PrefixExtractor.max_length, lowcase=True):
        """
        Creates a list with the most common prefixes found in wordlist.
        
        :param wordlist: a list of words (there shouldn't be repetitions)
        :param num: maximum number of prefixes
        :param min_occurrences: minimum number of occurrences of each prefix
        in wordlist
        :param length: desired length of prefixes
        :param lowcase: set the affix in lowercase
        """
        # we take the whole word if len(x) <= length
        lowcaser = lambda x: x.lower() if lowcase else x
        all_beginnings = (lowcaser(x[-length:]) for x in wordlist 
                       if  not re.search('_|\d', x[-length:]))
        c = Counter(all_beginnings)
        common_beginnings = c.most_common(num)
        return [e for e, n in common_beginnings if n >= min_occurrences]

# ----------------------------------------------------------------------

cdef class GazetteerExtractor(Extractor):

    absent = 0
    present = 1
    padding = 2
    num_values = 3              # extractor values

    # cdef bool lowcase
    # cdef bool noaccents

    def __init__(self, ngrams=None, size=5, lowcase=True, noaccents=True):
        """
        :param ngrams: iterator on ngrams (list of words) to add to gazeetteer.
        :param size: vector dimension.
        :param lowcase: whether to compare lowercase words.
        :param noaccents: whether to remove accents from words.
        """
        super(GazetteerExtractor, self).__init__()
        self.lowcase = lowcase
        self.noaccents = noaccents
        if ngrams:
            self.dict = <dict>Trie()
            for ngram in ngrams:
                self.dict.add(ngram)
            self.table = embeddings.generate_vectors(GazetteerExtractor.num_values, size)

    cpdef int_t get_padding_left(self):
        ":return: the feature representing the token used as left padding"
        return GazetteerExtractor.padding

    cpdef int_t get_padding_right(self):
        ":return: the feature representing the token used as right padding"
        return GazetteerExtractor.padding

    @classmethod
    def normalize(cls, w, lowcase, noaccents):
        if lowcase: w = w.lower()
        if noaccents: w = strip_accents(w)
        return w 

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Check presence in dictionary possibly as multiword.
        Set to 'present' items corresponding to tokens present in dictionary
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if -1.
        :return: the list of codes for the given :param tokens:.
        """
        res = [GazetteerExtractor.absent] * len(tokens)
        if field >= 0:
            words = [GazetteerExtractor.normalize(w[field], self.lowcase, self.noaccents) for w in tokens]
        else:
            words = [GazetteerExtractor.normalize(w, self.lowcase, self.noaccents) for w in tokens]
        for start, token in enumerate(words):
            for end in self.dict.iter(words, start, self.lowcase, self.noaccents):
                for k in range(start, end):
                    res[k] = GazetteerExtractor.present
        return res

    @classmethod
    def create(cls, filename, size, lowcase=True, noaccents=True):
        """
        Create extractors from gazeeteer file, consisting of lines:
          TYPE\tentity
        :param filename: file where to read list.
        :param size: size of vectors to generate.
        :param lowcase: whether to compare lowercase words.
        :param noaccents: whether to remove accents from words.
        """
        classes = OrderedDict() # preserve insertion order
        with open(filename) as file:
            for line in file:
                line = line.strip().decode('utf-8')
                c, words = line.split(None, 1)
                words = [cls.normalize(w, lowcase, noaccents) for w in words.split()]
                if c not in classes:
                    classes[c] = Trie()
                classes[c].add(words)
        extractors = [cls(trie, size, lowcase, noaccents) for trie in classes.values()]
        return extractors

    # min number of occurrences in corpus to put in gazetteer
    # FIXME: consider also stopping at punctuation
    minOccurr = 1

    @classmethod
    def build(cls, sentences, formField, tagField=-1, lowcase=True, noaccents=True):
        """
        Build a trie for each tag in :param sentences: which counts the
        frequency of each ngram with that tag.
        :param lowcase: whether to compare lowercase words.
        :param noaccents: whether to remove accents from words.
        """
        # gazetteers must be kept in the same order as tags
        tries = OrderedDict()
        # collect n-gram
        for sent in sentences:
            ngram = []
            prevTag = 'O'
            for tok in sent:
                tag = tok[tagField]
                if tag[0] == 'B' or tag[0] == 'S': # Begin or Single
                    # terminates previous ngram, start next one
                    form = tok[formField].lower() # lowercase
                    if ngram:
                        clas = prevTag[2:] # strip B-/I-
                        tries.setdefault(clas, Trie()).add(ngram, lowcase, noaccents)
                    ngram = [form]
                elif tag == 'O':
                    if ngram:              # terminated ngram
                        clas = prevTag[2:] # strip B-/I-
                        tries.setdefault(clas, Trie()).add(ngram, lowcase, noaccents)
                    ngram = []
                elif ngram:     # continuing ngram (I or E)
                    # assert(prevTag[2:] == tag[2:])
                    form = tok[formField].lower() # lowercase
                    ngram.append(form)
                prevTag = tag
            # leftover
            if ngram:
                clas = prevTag[2:] # strip B-/I-
                tries.setdefault(clas, Trie()).add(ngram, lowcase, noaccents)
        for trie in tries.itervalues():
            trie.prune(GazetteerExtractor.minOccurr)
        return tries

    def save(self, file):
        super(GazetteerExtractor, self).save(file)
        pickle.dump(self.lowcase, file)
        pickle.dump(self.noaccents, file)

    def load(self, file):
        self.dict = <dict>pickle.load(file)
        self.table = pickle.load(file)
        self.lowcase = pickle.load(file)
        self.noaccents = pickle.load(file)

# ----------------------------------------------------------------------

cdef class AttributeExtractor(Extractor):
    """
    Extract a token attribute as feature.
    """

    padding = 0

    def __init__(self, values, size=5):
        """
        :param values: set of attribute values.
        :param size: vector dimension.
        """
        super(AttributeExtractor, self).__init__()
        self.dict =  { x:i+1 for i,x in enumerate(values) }
        self.table = embeddings.generate_vectors(len(self.dict)+1, size)

    cpdef int_t get_padding_left(self):
        ":return: the feature representing the token used as left padding"
        return AttributeExtractor.padding

    cpdef int_t get_padding_right(self):
        ":return: the feature representing the token used as right padding"
        return AttributeExtractor.padding


# ----------------------------------------------------------------------

cdef class ScopeExtractor(Extractor):
    """Abstract class for scope extractors."""

    padding = 0
 
    COLUMNS = {
        'ID': 0,
        'FORM': 1,
        'LEMMA': 2,
        'CPOSTAG': 3,
        'POSTAG': 4,
        'FEATS': 5,
        'HEAD': 6,
        'DEPREL': 7,
        'PHEAD': 8,
        'PDEPREL': 9,
        'CUE': 10,
        'SCOPE': 11
    }

    UNKNOWN=-1
    NONE = -2

    def __init__(self, sentences, size=5):
        """
        :param values: set of attribute values.
        :param size: vector dimension.
        """
        super(ScopeExtractor, self).__init__()
    
        self.extract_dict(sentences) #assign self.dict
        self.dict[self.UNKNOWN] = len(self.dict)
        self.dict[self.NONE] = len(self.dict)
        self.table = embeddings.generate_vectors(len(self.dict)+1, size)

    
    cdef set _get_tokens_value(self, sentences, position):
        return set([tok[position] for (sent, tree) in sentences for tok in sent])

    # to override this method
    cdef dict extract_dict(self, sentences):
        self.dict = {}

    cpdef int_t get_padding_left(self):
        ":return: the feature representing the token used as left padding"
        return AttributeExtractor.padding

    cpdef int_t get_padding_right(self):
        ":return: the feature representing the token used as right padding"
        return AttributeExtractor.padding

# ----------------------------------------------------------------------

cdef class ScopeExtractorCandidateCueDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        values = [p for p in itertools.product(s, repeat=2)]
        self.dict = {'-'.join(v):i for i,v in enumerate(values)}

    
    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """        
        return [self.dict.get('%s-%s' % (other['node'].value[self.COLUMNS['DEPREL']], other['cue'].value[self.COLUMNS['DEPREL']]), self.dict[self.UNKNOWN])]

# ----------------------------------------------------------------------

cdef class ScopeExtractorCandidatePos(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """        
        return [self.dict.get(other['node'].value[ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorCandidateLemma(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #lemma_set = set([tok[ScopeExtractor.COLUMNS['LEMMA']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['LEMMA'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """        
        return [self.dict.get(other['node'].value[ScopeExtractor.COLUMNS['LEMMA']], self.dict[self.UNKNOWN])]

cdef class ScopeExtractorCandidateForm(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['FORM'])
        #form_set = set([tok[ScopeExtractor.COLUMNS['FORM']] for (sent, tree) in sentences for tok in sent])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """
        return [self.dict.get(other['node'].value[ScopeExtractor.COLUMNS['FORM']], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorCandidateDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """        
        return [self.dict.get(other['node'].value[ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorLeftCandidatePos(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        node_id = other['node'].value[ScopeExtractor.COLUMNS['ID']]
        if node_id == 1:
            return [self.dict[self.NONE]]
        else:
            return [self.dict.get(other['sentence'][node_id-2][ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.UNKNOWN])]

cdef class ScopeExtractorRightCandidatePos(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        node_id = other['node'].value[ScopeExtractor.COLUMNS['ID']]
        if node_id == len(other['sentence']):
            return [self.dict[self.NONE]]
        else:
            return [self.dict.get(other['sentence'][node_id][ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorLeftCandidateDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        node_id = other['node'].value[ScopeExtractor.COLUMNS['ID']]
        if node_id == 1:
            return [self.dict[self.NONE]]
        else:
            return [self.dict.get(other['sentence'][node_id-2][ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorRightCandidateDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        node_id = other['node'].value[ScopeExtractor.COLUMNS['ID']]
        if node_id == len(other['sentence']):
            return [self.dict [self.NONE]]
        else:
            return [self.dict.get(other['sentence'][node_id][ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorCandidateCueType(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        cue_set = set([tok[ScopeExtractor.COLUMNS['CUE']].split('(')[0] for (sent, tree) in sentences for tok in sent])
        self.dict = {v:i+1 for i,v in enumerate(cue_set)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """        
        return [self.dict.get(other['node'].value[ScopeExtractor.COLUMNS['CUE']].split('(')[0], self.dict[self.UNKNOWN])]


cdef class ScopeExtractorCandidateIsCue(ScopeExtractor):

    TRUE = 1
    FALSE = 2

    cdef dict extract_dict(self, sentences):
        self.dict = {True: self.TRUE, False: self.FALSE}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).
        """        
        return [self.dict.get(other['node'].value[ScopeExtractor.COLUMNS['CUE']] != 'O', self.dict[self.UNKNOWN])]


cdef class ScopeExtractorScopeLength(ScopeExtractor):

    MAX = 100

    cdef dict extract_dict(self, sentences):
        self.dict = {i:i+1 for i in xrange(self.MAX+1)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """ 
        return [self.dict.get(len(other['scope']), self.dict[self.MAX])]


cdef class ScopeExtractorCueCandidateDistance(ScopeExtractor):

    MAX = 100

    cdef dict extract_dict(self, sentences):
        self.dict = {i:i+1 for i in xrange(self.MAX+1)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """ 
        return [self.dict.get(abs(other['cue'].value[ScopeExtractor.COLUMNS['ID']]-other['node'].value[ScopeExtractor.COLUMNS['ID']]), self.dict[self.MAX])]


cdef class ScopeExtractorCueCandidateDistanceRange(ScopeExtractor):

    RANGE_0_4 = 1
    RANGE_5_10 = 2
    RANGE_11 = 3

    cdef dict extract_dict(self, sentences):
        self.dict = {
            self.RANGE_0_4: self.RANGE_0_4,
            self.RANGE_5_10: self.RANGE_5_10,
            self.RANGE_11: self.RANGE_11
        }

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope).

        """
        
        distance = abs(other['cue'].value[ScopeExtractor.COLUMNS['ID']]-other['node'].value[ScopeExtractor.COLUMNS['ID']])
        ret = self.dict[self.UNKNOWN]
        if 0 <= distance <= 4:
            ret = self.RANGE_0_4
        elif 5 <= distance <= 10:
            ret = self.RANGE_5_10
        elif distance >= 11:
            ret = self.RANGE_11
        
        return [self.dict[ret]]


cdef class ScopeExtractorLastDescendantPos(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """

        last_desc = None
        l = {}

        # check if lal or ral
        if other['node'].value[ScopeExtractor.COLUMNS['ID']] < other['cue'].value[ScopeExtractor.COLUMNS['ID']]:
            # lal - rightmost descendant
            rc = other['node'].descendants([other['node'].right])
            for c in rc:
                if c.value[ScopeExtractor.COLUMNS['ID']] > other['node'].value[ScopeExtractor.COLUMNS['ID']]:
                    l[c.value[ScopeExtractor.COLUMNS['ID']]] = c
            last_desc = l[max(l.keys())] if l else None
        else:
            # ral - leftmost descendant
            lc = other['node'].descendants([other['node'].left])
            for c in lc:
                if c.value[ScopeExtractor.COLUMNS['ID']] < other['node'].value[ScopeExtractor.COLUMNS['ID']]:
                    l[c.value[ScopeExtractor.COLUMNS['ID']]] = c
            last_desc = l[min(l.keys())] if l else None

        return [self.dict.get(last_desc.value[ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.UNKNOWN]) if last_desc else self.dict[self.NONE]]


cdef class ScopeExtractorLastDescendantDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}
        #self.dict[self.NONE] = len(self.dict)

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """

        last_desc = None
        l = {}

        # check if lal or ral
        if other['node'].value[ScopeExtractor.COLUMNS['ID']] < other['cue'].value[ScopeExtractor.COLUMNS['ID']]:
            # lal - rightmost descendant
            rc = other['node'].descendants([other['node'].right])
            for c in rc:
                if c.value[ScopeExtractor.COLUMNS['ID']] > other['node'].value[ScopeExtractor.COLUMNS['ID']]:
                    l[c.value[ScopeExtractor.COLUMNS['ID']]] = c
            last_desc = l[max(l.keys())] if l else None
        else:
            # ral - leftmost descendant
            lc = other['node'].descendants([other['node'].left])
            for c in lc:
                if c.value[ScopeExtractor.COLUMNS['ID']] < other['node'].value[ScopeExtractor.COLUMNS['ID']]:
                    l[c.value[ScopeExtractor.COLUMNS['ID']]] = c
            last_desc = l[min(l.keys())] if l else None

        return [self.dict.get(last_desc.value[ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.UNKNOWN]) if last_desc else self.dict[self.NONE]]


cdef class ScopeExtractorNextListPos(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        if other['next']:
            ret = self.dict.get(other['next'].value[ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.NONE])
        return [ret]


cdef class ScopeExtractorNextListDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        if other['next']:
            ret = self.dict.get(other['next'].value[ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.NONE])
        return [ret]


cdef class ScopeExtractorCandidateLeftSiblingDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}
        self.dict[self.NONE] = len(self.dict)

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        s = other['node'].getLeftSibling()
        if s:
            ret = self.dict.get(s.value[ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.NONE])
        return [ret]

cdef class ScopeExtractorCandidateRightSiblingDepRel(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['DEPREL']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['DEPREL'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        s = other['node'].getRightSibling()
        if s:
            ret = self.dict.get(s.value[ScopeExtractor.COLUMNS['DEPREL']], self.dict[self.NONE])
        return [ret]


cdef class ScopeExtractorCandidateLeftSiblingPos(ScopeExtractor):
    
    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        s = other['node'].getLeftSibling()
        if s:
            ret = self.dict.get(s.value[ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.NONE])
        return [ret]

cdef class ScopeExtractorCandidateRightSiblingPos(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['POSTAG']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['POSTAG'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        s = other['node'].getRightSibling()
        if s:
            ret = self.dict.get(s.value[ScopeExtractor.COLUMNS['POSTAG']], self.dict[self.NONE])
        return [ret]


cdef class ScopeExtractorCandidateLeftSiblingLemma(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #pos_set = set([tok[ScopeExtractor.COLUMNS['LEMMA']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['LEMMA'])
        self.dict = {v:i+1 for i,v in enumerate(s)}

    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        s = other['node'].getLeftSibling()
        if s:
            ret = self.dict.get(s.value[ScopeExtractor.COLUMNS['LEMMA']], self.dict[self.NONE])
        return [ret]

cdef class ScopeExtractorCandidateRightSiblingLemma(ScopeExtractor):

    cdef dict extract_dict(self, sentences):
        #deprel_set = set([tok[ScopeExtractor.COLUMNS['LEMMA']] for (sent, tree) in sentences for tok in sent])
        s = self._get_tokens_value(sentences, ScopeExtractor.COLUMNS['LEMMA'])
        self.dict = {v:i+1 for i,v in enumerate(s)}
        
    cpdef extract(self, list tokens, int_t field, dict other):
        """
        Extract the features representing each token.
        :param tokens: list of tokens.
        :param field: which token field to use, the whole token if None.
        :param other: a dictionary with other params (tree, cue, node, scope, sentence, next).

        """
        ret = self.dict[self.NONE]
        s = other['node'].getRightSibling()
        if s:
            ret = self.dict.get(s.value[ScopeExtractor.COLUMNS['LEMMA']], self.dict[self.NONE])
        return [ret]
