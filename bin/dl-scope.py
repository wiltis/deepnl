#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and use a Scope Detection Tagger

Author: Daniele Sartiano
"""

import logging
import numpy as np
import argparse
from ConfigParser import ConfigParser

# allow executing from anywhere without installing the package
import sys
import os
import distutils.util
builddir = os.path.dirname(os.path.realpath(__file__)) + '/../build/lib.'
libdir = builddir + distutils.util.get_platform() + '-' + '.'.join(map(str, sys.version_info[:2]))
sys.path.insert(0, libdir)      # DEBUG

# local
from deepnl.reader import ScopeReader
from deepnl.corpus import ConllWriter
from deepnl.extractors import *
from deepnl.trainer import ScopeTrainer
from deepnl.network import Network
from deepnl.networkconv import ConvolutionalNetwork
from deepnl.trainerconv import ConvTrainer
from deepnl.classifier import ScopeClassifier

# ----------------------------------------------------------------------
# Auxiliary functions

def create_trainer(args, converter, labels):
    """
    Creates or loads a neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = ScopeTrainer.load(args.load)
        # change learning rate
        trainer.learning_rate = args.learning_rate
        trainer.threads = args.threads
    else:
        logger.info('Creating new network...')
        # sum the number of features in all tables 
        feat_size = converter.size()
        pool_size = args.window
        
        input_size = converter.size() * args.window
        nn = Network(input_size, args.hidden, len(labels))
        options = {
            'learning_rate': args.learning_rate,
            'eps': args.eps,
            'ro': args.ro,
            'verbose': args.verbose,
            'left_context': args.window/2,
            'right_context': args.window/2
        }

        trainer = ScopeTrainer(nn, converter, labels, options)
        
    trainer.saver = saver(args.model, args.vectors, args.variant)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())
    
    return trainer

def saver(model_file, vectors_file, variant):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.save_vectors(vectors_file, variant)
        with open(model_file, 'wb') as file:
            trainer.classifier.save(file)
    return save


# ScopeExtractorCueCandidateDistance
extractors_dict = {
    'candidateCue': [
        ScopeExtractorCandidateCueDepRel,
        ScopeExtractorCandidateCueType,
        ScopeExtractorCueCandidateDistanceRange
    ],
    'candidate': [
        ScopeExtractorCandidatePos,
        ScopeExtractorCandidateLemma,
        ScopeExtractorCandidateForm,
        ScopeExtractorCandidateDepRel,
        ScopeExtractorCandidateIsCue
    ],
    'left_right': [
        ScopeExtractorLeftCandidatePos,
        ScopeExtractorRightCandidatePos,
        ScopeExtractorLeftCandidateDepRel,
        ScopeExtractorRightCandidateDepRel
    ],
    'last_desc': [
        ScopeExtractorLastDescendantPos,
        ScopeExtractorLastDescendantDepRel
    ],
    'next_list': [
        ScopeExtractorNextListPos,
        ScopeExtractorNextListDepRel
    ],
    'scope': [
        ScopeExtractorScopeLength
    ],
    'siblings': [
        ScopeExtractorCandidateLeftSiblingDepRel,
        ScopeExtractorCandidateRightSiblingDepRel,
        ScopeExtractorCandidateLeftSiblingPos,
        ScopeExtractorCandidateRightSiblingPos,
        ScopeExtractorCandidateLeftSiblingLemma,
        ScopeExtractorCandidateRightSiblingLemma

    ],
    'test': [
        ScopeExtractorCandidateSubtreeNodes
    ]

}

def conf_extractors(keys):
    return reduce(lambda x, y: x+y, [extractors_dict[k] for k in keys])

mapConfigurationExtractors = {
    0: conf_extractors(['candidateCue']),
    1: conf_extractors(['candidateCue', 'candidate']),
    2: conf_extractors(['candidateCue', 'candidate', 'left_right']),
    3: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc']),
    4: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc', 'siblings']),
    5: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc', 'siblings', 'next_list']),
    6: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc', 'siblings', 'next_list', 'scope']),
    7: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc', 'siblings', 'scope']),
    8: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc', 'siblings', 'scope', 'test']),
    9: conf_extractors(['candidateCue', 'candidate', 'left_right', 'last_desc', 'siblings', 'next_list', 'scope', 'test'])
}

# ----------------------------------------------------------------------

def main():

    # set the seed for replicability
    np.random.seed(89) #(42)

    defaults = {}
    
    parser = argparse.ArgumentParser(description="Train or use a Scope Detection Tagger.")
    
    parser.add_argument('-c', '--config', dest='config_file',
                        help='Specify config file', metavar='FILE')

    parser.add_argument('model', type=str,
                        help='Model file to train/use.')

    # training options
    train = parser.add_argument_group('Train')
    train.add_argument('-t', '--train', type=str, default='',
                        help='File with annotated data for training.')
    train.add_argument('-w', '--window', type=int, default=5,
                        help='Size of the word window (default %(default)s)')
    train.add_argument('-s', '--embeddings-size', type=int, default=50,
                        help='Number of features per word (default %(default)s)',
                        dest='embeddings_size')
    train.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs (default %(default)s)',
                        dest='iterations')
    train.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Learning rate for network weights (default %(default)s)',
                        dest='learning_rate')
    train.add_argument('-n', '--hidden', type=int, default=200,
                        help='Number of hidden neurons (default %(default)s)',
                        dest='hidden')
    train.add_argument('--eps', type=float, default=1e-6,
                        help='Epsilon value for AdaGrad (default %(default)s)')
    train.add_argument('--ro', type=float, default=0.95,
                        help='Ro value for AdaDelta (default %(default)s)')
    train.add_argument('-o', '--output', type=str, default='',
                        help='File where to save embeddings')

    # Embeddings
    embeddings = parser.add_argument_group('Embeddings')
    embeddings.add_argument('--vocab', type=str, default='',
                        help='Vocabulary file, either read or created')
    embeddings.add_argument('--vocab-size', type=int, default=0,
                            help='Maximum size of vocabulary from corpus (default %(default)s)')
    embeddings.add_argument('--vectors', type=str, default='',
                        help='Embeddings file, either read or created')
    embeddings.add_argument('--min-occurr', type=int, default=3,
                        help='Minimum occurrences for inclusion in vocabulary (default %(default)s',
                        dest='minOccurr')
    embeddings.add_argument('--load', type=str, default='',
                        help='Load previously saved model')
    embeddings.add_argument('--variant', type=str, default='',
                        help='Either "senna" (default), "polyglot" or "word2vec".')

    # Extractors:
    extractors = parser.add_argument_group('Extractors')
    extractors.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                        help='Include capitalization features. Optionally, supply the number of features (default %(default)s)')
    extractors.add_argument('--pos', const=1, type=int, nargs='?', default=None,
                        help='Use POS tag. Optionally supply the POS token field index (default %(default)s)')
    extractors.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                            help='Include suffix features. Optionally, supply the number of features (default %(default)s)')
    extractors.add_argument('--suffixes', type=str, default='',
                        help='Load suffixes from this file')
    extractors.add_argument('--prefix', const=5, nargs='?', type=int, default=None,
                            help='Include prefix features. Optionally, '\
                            'supply the number of features (default %(default)s)')
    extractors.add_argument('--prefixes', type=str, default='',
                        help='Load prefixes from this file')
    extractors.add_argument('--gazetteer', type=str,
                        help='Load gazetteer from this file')
    extractors.add_argument('--gsize', type=int, default=5,
                        help='Size of gazetteer features (default %(default)s)')

    # Added a conf
    extractors.add_argument('--scope-feats-conf', type=int, default=1,
                            help='The scope feats configuration', dest='scopeFeatsConf')


    # reader
    parser.add_argument('--form-field', type=int, default=1,
                        help='Token field containing form (default %(default)s)',
                        dest='formField')

    # common
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads (default %(default)s)')
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        action='store_true')

    # Use this for obtaining defaults from config file:
    args = parser.parse_args()


    # logger settings
    log_format = '%(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format=log_format, level=log_level)
    logger = logging.getLogger("Logger")

    
    
    config = ConfigParser()
    if args.config_file:
        config.read(args.config_file)
    # merge args with config

    if args.train:
        reader = ScopeReader(args.formField)

        # a generator (can be iterated several times)
        sentence_iter = reader.read(args.train)

        if args.vocab and os.path.exists(args.vocab):
            if args.vectors and os.path.exists(args.vectors):
                # use supplied embeddings
                embeddings = Embeddings(vectors=args.vectors, vocab_file=args.vocab,
                                        variant=args.variant)
            else:
                # create random embeddings
                embeddings = Embeddings(args.embeddings_size, vocab_file=args.vocab,
                                        variant=args.variant)
            # add the ngrams from the corpus
            # build vocabulary and tag set
            if args.vocab_size:
                vocab = reader.create_vocabulary(sentence_iter,
                                                 args.vocab_size,
                                                 args.minOccurr)
                embeddings.merge(vocab)
                logger.info("Overriding vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)


        elif args.variant == 'word2vec':
            if os.path.exists(args.vectors):
                embeddings = Embeddings(vectors=args.vectors,
                                        variant=args.variant)
                vocab = reader.create_vocabulary(sentence_iter,
                                                 args.vocab_size,
                                                 args.minOccurr)
                embeddings.merge(vocab)
            else:
                vocab = reader.create_vocabulary(sentence_iter,
                                                 args.vocab_size,
                                                 args.minOccurr)
                embeddings = Embeddings(vocab=vocab,
                                        variant=args.variant)
            if args.vocab:
                logger.info("Saving vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)
        elif not args.vocab_size:
            logger.error("Missing parameter --vocab-size")
            return
            
        else:
            # build vocabulary
            vocab = reader.create_vocabulary(sentence_iter,
                                             args.vocab_size,
                                             args.minOccurr)
            logger.info("Creating word embeddings")
            embeddings = Embeddings(args.embeddings_size, vocab=vocab,
                                    variant=args.variant)
            if args.vocab:
                logger.info("Saving vocabulary in %s" % args.vocab)
                embeddings.save_vocabulary(args.vocab)

        converter = Converter()
        # pass just the formField from tokens to the extractor
        converter.add(embeddings, reader.formField)

        # TODO: add choice
        logger.info("Creating Scope Extractors...")
        for fun in mapConfigurationExtractors[args.scopeFeatsConf]:
            converter.add(fun(sentence_iter))

        # if args.scopeFeatsConf == 1:
        #     logger.info("Creating Scope Extractors...")

        #     converter.add(ScopeExtractorCandidateCueDepRel(sentence_iter))
        #     converter.add(ScopeExtractorCandidatePos(sentence_iter))
        #     converter.add(ScopeExtractorCandidateLemma(sentence_iter))
        #     converter.add(ScopeExtractorCandidateForm(sentence_iter))
        #     converter.add(ScopeExtractorCandidateDepRel(sentence_iter))
        #     converter.add(ScopeExtractorCandidateIsCue(sentence_iter))
        #     converter.add(ScopeExtractorCandidateCueType(sentence_iter))


        #     converter.add(ScopeExtractorLeftCandidatePos(sentence_iter))
        #     converter.add(ScopeExtractorRightCandidatePos(sentence_iter))
        #     converter.add(ScopeExtractorLeftCandidateDepRel(sentence_iter))
        #     converter.add(ScopeExtractorRightCandidateDepRel(sentence_iter))
            
        #     #converter.add(ScopeExtractorCueCandidateDistance(sentence_iter))
        #     converter.add(ScopeExtractorCueCandidateDistanceRange(sentence_iter))
            
        #     converter.add(ScopeExtractorLastDescendantPos(sentence_iter))
        #     converter.add(ScopeExtractorLastDescendantDepRel(sentence_iter))
            
        #     converter.add(ScopeExtractorNextListPos(sentence_iter))
        #     converter.add(ScopeExtractorNextListDepRel(sentence_iter))

        #     converter.add(ScopeExtractorScopeLength(sentence_iter))

        examples = []

        labels = []
        labels_index = {}
        labels_ids = []
        for i, c in enumerate([-1,1]):
            labels_index[c] = i
            labels.append(c)
        
        
        #extract examples (a list of scopes) from train set
        for i, (sent, tree) in enumerate(sentence_iter):
            if not tree:
                continue
            cues = reader.getCues(tree)
            
            for cue in cues:

                scope = set([cue])
                
                visited = []
                r_ral = [cue]
                cue.ral(visited, r_ral)

                for ii, n in enumerate(r_ral[1:]):
                    label = 1 if reader.node_in_scope(cue, n) else -1
                    other = {
                        'tree': tree,
                        'cue': cue,
                        'node': n,
                        'scope': list(scope)[:],
                        'sentence': sent,
                        'next': r_ral[1:][ii+1] if len(r_ral[1:]) > ii+1 else None
                    }
                    
                    examples.append(converter.convert([sent[n.value[ScopeReader.COLUMNS['ID']]-1]], other))
                    labels_ids.append(labels_index[label])
                    if label == 1:
                        scope.add(n)
                        lc = n.descendants([n.left])
                        for c in lc:
                            if c.value[ScopeReader.COLUMNS['ID']] < n.value[ScopeReader.COLUMNS['ID']]:
                                scope.add(c)

                    else:
                        break
                                
                visited = []
                r_lal = [cue]
                cue.lal(visited, r_lal)

                for ii, n in enumerate(r_lal[1:]):
                    label = 1 if reader.node_in_scope(cue, n) else -1
                    other = {
                        'tree': tree,
                        'cue': cue,
                        'node': n,
                        'scope': list(scope)[:],
                        'sentence': sent,
                        'next': r_lal[1:][ii+1] if len(r_lal[1:]) > ii+1 else None
                    }

                    examples.append(converter.convert([sent[n.value[ScopeReader.COLUMNS['ID']]-1]], other))
                    labels_ids.append(labels_index[label])
                    
                    if label == 1:
                        scope.add(n)
                        rc = n.descendants([n.right])
                        for c in rc:
                            if c.value[ScopeReader.COLUMNS['ID']] > n.value[ScopeReader.COLUMNS['ID']]:
                                scope.add(c)

                    else:
                        break

        logger.info("Vocabulary size: %d" % embeddings.dict.size())
        logger.info("Tagset size: %d" % len(labels))
        trainer = create_trainer(args, converter, labels)
        logger.info("Starting training with %d sentences" % len(examples))

        report_frequency = max(args.iterations / 200, 1)
        report_frequency = 1    # DEBUG
        trainer.train(examples, labels_ids, args.iterations, report_frequency, args.threads)
    
        logger.info("Saving trained model ...")
        trainer.saver(trainer)
        logger.info("... to %s" % args.model)

    else:
        with open(args.model) as file:
            classifier = ScopeClassifier.load(file)
        reader = ScopeReader(1)
        sentence_iter = reader.read()

        for i, (sent, tree) in enumerate(sentence_iter):
            if tree:
                cues = reader.getCues(tree)
            else:
                cues = []

            labels = ['_'] * len(sent)
            for cue in cues:
                scope = set([cue])
                visited = []
                r_ral = [cue]                    
                cue.ral(visited, r_ral)
                for ii, n in enumerate(r_ral[1:]):
                    other = {
                        'tree': tree,
                        'cue': cue,
                        'node': n,
                        'scope': list(scope)[:],
                        'sentence': sent,
                        'next': r_ral[1:][ii+1] if len(r_ral[1:]) > ii+1 else None
                    }

                    label = classifier.predict([sent[n.value[ScopeReader.COLUMNS['ID']]-1]], other)
                    #print 'the label is (ral)', label                    
                    if label == 1:
                        scope.add(n)
                        lc = n.descendants([n.left])
                        for c in lc:
                            if c.value[ScopeReader.COLUMNS['ID']] < n.value[ScopeReader.COLUMNS['ID']]:
                                scope.add(c)
                    else:
                        break

                visited = []
                r_lal = [cue]
                cue.lal(visited, r_lal)

                for ii, n in enumerate(r_lal[1:]):
                    other = {
                        'tree': tree,
                        'cue': cue,
                        'node': n,
                        'scope': list(scope)[:],
                        'sentence': sent,
                        'next': r_lal[1:][ii+1] if len(r_lal[1:]) > ii+1 else None
                    }
                    label = classifier.predict([sent[n.value[ScopeReader.COLUMNS['ID']]-1]], other)
                    #print 'the label is (lal)', label
                    if label == 1:
                        scope.add(n)
                        rc = n.descendants([n.right])
                        for c in rc:
                            if c.value[ScopeReader.COLUMNS['ID']] > n.value[ScopeReader.COLUMNS['ID']]:
                                scope.add(c)
                    else:
                        break

                scope_cue_id = cue.value[ScopeReader.COLUMNS['CUE']].split('(')[1][:-1]

                for node in scope:
                    index = node.value[ScopeReader.COLUMNS['ID']]-1
                    if labels[index] != '_':
                        labels[index] = '%s,%s' % (scope_cue_id, labels[index])
                    else:
                        labels[index] = scope_cue_id

            
            for i, token in enumerate(sent):
                if labels[i] == '_' and token[ScopeReader.COLUMNS['CUE']].startswith('B-'):
                    labels[i] = token[ScopeReader.COLUMNS['CUE']].split('(')[1][:-1]
                out = '%s\t%s' % ('\t'.join([unicode(f) for f in token[:-1]]), labels[i])
                print >> sys.stdout, out.encode('utf-8')
            print >> sys.stdout

# ----------------------------------------------------------------------

profile = False

if __name__ == '__main__':
    if profile:
        import cProfile
        cProfile.runctx("main()", globals(), locals(), "Profile.prof")
    else:
        main()
