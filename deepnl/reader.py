#!/usr/env python
# -*- coding: utf-8 -*-
#cython: embedsignature=True

"""
Classes for reading various types of corpora.
"""

# standard
import os
import logging
import numpy as np
from collections import Counter
import gzip

# local
from corpus import *
from embeddings import Plain

class Reader(object):
    """
    Abstract class for corpus readers.
    """
    
    # force class to be abstract
    #__metaclass__ = abc.ABCMeta

    def create_vocabulary(self, sentences, size, min_occurrences=3):
        """
        Create vocabulary from sentences.
        :param sentences: an iterable on sentences.
        :param size: size of the vocabulary
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        Sentence tokens are lists [form, ..., tag]
        """
        c = Counter()
        for sent in sentences:
            for token, in sent:
                c[token] += 1
        common = c.most_common(size)
        words = [w for w, n in common if n >= min_occurrences]
        return words

    def load_vocabulary(self, filename):
        return Plain.read_vocabulary(filename)

# ----------------------------------------------------------------------

class TextReader(Reader):
    """
    Reads sentences from tokenized text file.
    """
    
    def __init__(self, variant=None):
        """
        :param sentences: A list of lists of tokens.
        """
        super(TextReader, self).__init__()
        self.variant = variant

    def read(self, filename=None):
        """
        :param filename: name of the file from where sentences are read.
            The file should have one sentence per line, with tokens
            separated by white spaces.
        :return: an iterable over sentences, which can be iterated over several
            times.
        """
        class iterable(object):
            def __iter__(self):
                if not filename:
                    file = sys.stdin
                elif filename.endswith('.gz'):
                    file = gzip.GzipFile(filename, 'rb')
                else:
                    file = open(filename, 'rb')
                for line in file:
                    sent =  unicode(line, 'utf-8').split()
                    if sent:
                        yield sent
                file.close()

        return iterable()

    def sent_count(self):
        return len(self.sentences)

# ----------------------------------------------------------------------

class TaggerReader(ConllReader):
    """
    Abstract class extending TextReader with useful functions
    for tagging tasks. 
    """
    
    # force class to be abstract
    #__metaclass__ = abc.ABCMeta

    def __init__(self, formField=0, tagField=-1):
        """
        :param formField: the position of the form field in tokens
        :param tagField: the position of the tag field in tokens
        """
        super(TaggerReader, self).__init__()
        # self.sentence_count = len(sentences) if sentences else 0
        self.formField = formField # field containing form
        self.tagField = tagField # field containing tag

    def read(self, filename):
        """
        :return: an iterator on sentences.
        """
        return ConllReader(filename)

    # def sent_count(self):
    #     return self.sentence_count

    def create_vocabulary(self, sentences, size, min_occurrences=3):
        """
        Create vocabulary and tag set from sentences.
        :param sentences: an iterable on sentences.
        :param size: size of the vocabulary
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        Sentence tokens are lists [form, ..., tag]
        """
        c = Counter()
        tags = set()
        for sent in sentences:
            for token in sent:
                c[token[self.formField]] += 1
                tags.add(token[self.tagField])
        common = c.most_common(size) # common is a list of pairs
        words = [w for w, n in common if n >= min_occurrences]
        return words, tags

    def create_tagset(self, sentences):
        """
        Create tag set from sentences.
        :param sentences: an iterable over sentences.
        """
        tags = set()
        for sent in sentences:
            for token in sent:
                tags.add(token[self.tagField])
        return tags
    
# ----------------------------------------------------------------------

class PosReader(TaggerReader):
    """
    This class reads data from a POS corpus and turns it into a representation
    for use by the neural network for the POS tagging task.
    """
    
    def __init__(self, formField=0, tagField=-1):
        self.rare_tag = None
        super(PosReader, self).__init__(formField=0, tagField=-1)

# ----------------------------------------------------------------------

class TweetReader(Reader):
    """
    Reader for tweets in SemEval 2013 format, one tweet per line consisting  of:
    SID	UID	polarity	tokenized text
    264183816548130816      15140428        positive      Gas by my house hit $3.39!!!! I'm going to Chapel Hill on Sat. :)
    """

    def __init__(self, text_field=3, label_field=2, ngrams=1, variant=None):
        """
	:param ngrams: the length of ngrams to consider.
	:param variant: whether to use native, or SENNA or Polyglot conventions.
        """
        super(TweetReader, self).__init__()
        self.text_field = text_field
        self.label_field = label_field
	self.ngrams = ngrams
        self.variant = variant
        self.sentences = []
        self.polarities = []

    def __iter__(self):
        for tweet in TsvReader(): # stdin
            yield tweet

    def read(self, filename=None):
        """
        Builds a list of sentences and a list corresponding polarities [-1, 0, 1]
        """
        for tweet in TsvReader(filename):
            if len(tweet) <= self.text_field:
                # drop empty tweets
                continue
            if tweet[self.label_field] == 'positive':
                polarity = 1
            elif tweet[self.label_field] == 'negative':
                polarity = -1
            else:
                polarity = 0    # neutral or objective
            self.sentences.append(tweet[self.text_field].split())
            self.polarities.append(polarity)
        return self.sentences
                    
    def acceptable(self, token):
        """Simple criteron to accept a token as part of a phrase, rejecting
        punctuations or common short words.
        """
        return len(token) > 2

    # discount to avoid phrases with very infrequent words
    delta = 1

    def create_vocabulary(self, tweets, size=None, min_occurrences=3, threshold=0.1):
        """
        Generates a list of all ngrams from the given tweets.
        
        :param tweets: an iterable on tweets.
        :param size: Max number of tokens to be included in the dictionary.
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        :param threshold: minimum bigram score.
        :return: list of ngrams (joined by '_'), list of bigrams
            and list of trigrams.
        """
        
        # Use PMI-like score for selecting collocations:
        # score(x, y) = (count(x,y) - delta) / count(x)count(y)
        # @see Mikolov et al. 2013. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013
        # unigrams
        unigramCount = Counter(token for tweet in tweets for token in tweet)
        ngrams = [u for u,c in unigramCount.iteritems() if c >= min_occurrences]
        # bigrams
        bigramCount = Counter()
        trigramCount = Counter()
        for tweet in tweets:
            for a,b,c in zip(tweet[:-1], tweet[1:], tweet[2:]):
                if unigramCount[a] >= min_occurrences and unigramCount[b] >= min_occurrences:
                    bigramCount.update([(a, b)])
                    if unigramCount[c] >= min_occurrences:
                        trigramCount.update([(a, b, c)])
        if len(tweet) > 1 and unigramCount[tweet[-2]] >= min_occurrences and unigramCount[tweet[-1]] >= min_occurrences:
            bigramCount.update([(tweet[-2], tweet[-1])])
        bigrams = []
        for b, c in bigramCount.iteritems():
            if (float(c) - TweetReader.delta) / (unigramCount[b[0]] * unigramCount[b[1]]) > threshold:
                ngrams.append(b[0] + '_' + b[1])
                bigrams.append(b)
        trigrams = []
        for b, c in trigramCount.iteritems():
            if (float(c) - TweetReader.delta) / (unigramCount[b[0]] * unigramCount[b[1]]) > threshold/2 \
                and (float(c) - TweetReader.delta) / (unigramCount[b[1]] * unigramCount[b[2]]) > threshold/2:
                ngrams.append(b[0] + '_' + b[1] + '_' + b[2])
                trigrams.append(b)
        # FIXME: repeat for multigrams
        return ngrams, bigrams, trigrams

# ----------------------------------------------------------------------

class ClassifyReader(TweetReader):
    """
    Variant of TweetReader with multiple labels.
    """

    def read(self, filename=None):
        """
        Builds a list of sentences.
        """
        for tweet in TsvReader(filename):
            self.sentences.append(tweet[self.text_field].split())
            self.polarities.append(tweet[self.label_field])
        return self.sentences


# ----------------------------------------------------------------------

class ScopeReader(Reader):
    """
    This class reads data from a CoNLL corpus  and turns it into a representation
    for use by the neural network for the Scope Detection tagging task.
    """

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

    def __init__(self, formField=0, tagField=-1):
        super(ScopeReader, self).__init__()
        self.formField = self.COLUMNS['FORM']
        self.filename = None;

    def __iter__(self):
        if self.filename:
            file = codecs.open(self.filename, 'r', 'utf-8', errors='ignore')
        else:
            file = codecs.getreader('utf-8')(sys.stdin)
        sent = []
        for line in file:
            line = line.strip()
            if line:
                sent.append(line.split('\t'))
            else:
                yield (sent, self.getTree(sent))
                sent = []
        if sent:                # just in case
            yield (sent, self.getTree(sent))
        if self.filename:
            file.close()


    def read(self, filename=None):
        """
        :return: an iterator on sentences.
        """
        self.filename = filename
        return self


    def create_vocabulary(self, sentences, size, min_occurrences=3):
        """
        Create vocabulary from sentences.
        :param sentences: an iterable on sentences.
        :param size: size of the vocabulary
        :param min_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        Sentence tokens are lists [form, ..., tag]
        """
        c = Counter()
        for (sent, tree) in sentences:
            for token in sent:
                c[token[ScopeReader.COLUMNS['FORM']]] += 1
        common = c.most_common(size)
        words = [w for w, n in common if n >= min_occurrences]
        return words


    @staticmethod
    def getTree(sentence):
        ID = ScopeReader.COLUMNS['ID']
        HEAD = ScopeReader.COLUMNS['HEAD']

        children = {}
        for token in sentence:
            token[ID] = int(token[ID])
            token[HEAD] = int(token[HEAD])
            if not token[HEAD] in children:
                children[token[HEAD]] = []
            children[token[HEAD]].append(token)

        # Bad CoNLL
        if 0 not in children:
            print >> sys.stderr, 'BAD CONL: no 0' # DEBUG
            return None
        if len(children[0]) > 1:
            print >> sys.stderr, 'BAD CONLL: children[0] > 1' # DEBUG
            return None

        root = Node(children[0][0])
        if len(sentence) == 1:
            return root

        toEvaluate = [root.value[ID]]

        while toEvaluate:
            father = toEvaluate.pop(0)
            nn = root.find(father)
            for c in children[father]:
                nn.insertChild(c)
                if c[ID] in children:
                    toEvaluate.append(c[ID])

        return root
    

    @staticmethod
    def getCues(tree):
        return [n for n in tree.traverse() if n.value[ScopeReader.COLUMNS['CUE']].startswith('B')]

    @staticmethod
    def node_in_scope(cue, node):
        scopes = node.value[ScopeReader.COLUMNS['SCOPE']].split(',')
        scope_cue_id = cue.value[ScopeReader.COLUMNS['CUE']].split('(')[1][:-1]
        return scope_cue_id in scopes

    @staticmethod
    def tokens_in_scope(cue, tokens):
        labels = []
        scope_cue_id = cue.value[ScopeReader.COLUMNS['CUE']].split('(')[1][:-1]
        for token in tokens:
            scopes = token[ScopeReader.COLUMNS['SCOPE']].split(',')
            l = 1 if scope_cue_id in scopes else -1
            labels.append(l)
        return labels


class Node(object):
    """
    A node of a parse tree.
    """
    def __init__(self, value, parent=None):
        self.parent = parent
        self.value = value
        self.left = []
        self.right = []

    def insertChild(self, value):
        w = self.left if value[ScopeReader.COLUMNS['ID']] < self.value[ScopeReader.COLUMNS['ID']] else self.right
        w.append(Node(value, self))


    def isLeftMostChild(self):
        if self.parent and self.parent.left:
            return self == self.parent.left[0]
        return False

    def isRightMostChild(self):
        if self.parent and self.parent.right:
            return self == self.parent.right[-1]
        return False

    def find(self, value):
        if value == self.value[ScopeReader.COLUMNS['ID']]:
            return self
        else:
            for lr in [self.left, self.right]:
                for ch in lr:
                    val = ch.find(value)
                    if val:
                        return val
        return None

    def getSiblings(self):
        childs = []
        if self.parent:
            # the node is right child of the parent
            if self.parent.value[ScopeReader.COLUMNS['ID']] < self.value[ScopeReader.COLUMNS['ID']]:
                childs = self.parent.right
            else:
                childs = self.parent.left
        llist = []
        rlist = []
        l = llist
        for ch in childs:
            if ch == self:
                l = rlist
                continue
            l.append(ch)
        return llist, rlist
                

    def getLeftSibling(self):
        lsiblings, _ = self.getSiblings()
        return lsiblings[-1] if lsiblings else None

    def getRightSibling(self):
        _, rsiblings = self.getSiblings()
        return rsiblings[0] if rsiblings else None
                
                
    # debug
    def __repr__(self, level=0):
        p = self.parent.value[ScopeReader.COLUMNS['ID']] if self.parent else 0 
        ret = '%s%s %s - SCOPE(%s)\n' % ('\t'*level, self.value[ScopeReader.COLUMNS['ID']], self.value[ScopeReader.COLUMNS['FORM']], self.value[ScopeReader.COLUMNS['SCOPE']])
        
        for lr in [self.left, self.right]:
            for child in lr:
                ret += child.__repr__(level+1)
        return ret
        
    def traverse(self):
        ret = [self]
        for lr in [self.left, self.right]:
            for child in lr:
                ret.extend(child.traverse())
        return ret

    def descendants(self, where):
        ret = []
        for lr in where:
            for child in lr:
                ret.extend(child.descendants([child.left, child.right]))

        ret.insert(0, self)
        return list(set(ret))
        

    def ral(self, visited, result):
        visited.append(self)
        if self.value[0] > result[-1].value[0]:
            result.append(self)
        for rc in self.right:
            if rc not in visited:
                rc.ral(visited, result)
        if self.parent and self.parent not in visited:
            self.parent.ral(visited, result)

    def lal(self, visited, result):
        visited.append(self)
        if self.value[0] < result[-1].value[0]:
            result.append(self)
        for lc in sorted(self.left, key=lambda x: x.value[ScopeReader.COLUMNS['ID']], reverse=True):# righmost order
            if lc not in visited:
                lc.lal(visited, result)
        if self.parent and self.parent not in visited:
            self.parent.lal(visited, result)


    # not used
    def visit(self, visited):
        visited.append(self.value[ScopeReader.COLUMNS['ID']])
        yield self
        for lc in sorted(self.left, key=lambda x: x.value[ScopeReader.COLUMNS['ID']], reverse=True): #rightmost order
            if lc.value[ScopeReader.COLUMNS['ID']] not in visited:
                for x in lc.visit(visited):
                    yield x
        for rc in self.right:
            if rc.value[ScopeReader.COLUMNS['ID']] not in visited:
                for x in rc.visit(visited):
                    yield x
        if self.parent and self.parent.value[ScopeReader.COLUMNS['ID']] not in visited:
            for x in self.parent.visit(visited):
                yield x
