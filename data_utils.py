# -*- coding: utf-8 -*-
""" BERT classification fine-tuning: utilities to work with QPM tasks. """

import logging
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """ A single training/test example for question pairs matching task. """
    def __init__(self, guid, question_a, question_b, label=None, category=None, hand_features=None):
        """ Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            question_a: string. The untokenized question sentence of the first sequence.
            question_b: string. The untokenized question sentence of the second sequence.
            label: string. The label of the example. This should be specified for train and dev examples,
            but not for test examples
        """
        self.guid = guid
        self.question_a = question_a
        self.question_b = question_b
        self.label = label
        self.category = category
        self.hand_features = hand_features


class InputFeatures(object):
    """ A single set of features of data. """
    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 category_clf_input_ids, category_clf_input_mask, category_clf_segment_ids, category_id, hand_features):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.category_clf_input_ids = category_clf_input_ids
        self.category_clf_input_mask = category_clf_input_mask
        self.category_clf_segment_ids = category_clf_segment_ids
        self.category_id = category_id
        self.hand_features = hand_features


class DataProcessor(object):
    """ Base class for data converters for sequence classfication data sets. """
    def get_examples(self, data_dir, set_type):
        """ Gets a collection of `InputExample`s for the train set. """
        raise NotImplementedError()

    def get_labels(self):
        """ Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """ Reads a `,` seperated value file. """
        with open(input_file, 'r', encoding='utf-8') as f:
            print(input_file)
            df = pd.read_csv(f, delimiter=',')
            df_feat = pd.read_csv(input_file.replace('.csv', '_feats.csv'))
            lines = []
            for index in df.index:
                line = df.iloc[index].values
                line = line.tolist()
                line.append(df_feat.iloc[index])
                lines.append(line)
            return lines


class QPMProcessor(DataProcessor):
    """ Processor for the QPM data set. """
    def get_examples(self, data_dir, set_type):
        """ See base class. """
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, set_type + '.csv')), set_type)

    def get_labels(self):
        """ See base class. """
        return [0, 1]

    def get_categories(self):
        return ['aids', 'hypertension', 'hepatitis', 'diabetes', 'breast_cancer', 'wrong']

    def _create_examples(self, lines, set_type):
        """ Creates examples for the training and dev sets. """
        examples = []

        for (i, line) in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            try:
                if set_type == 'train' or set_type == 'dev':
                    question_a = line[0]
                    question_b = line[1]
                    label = line[2]
                    category = line[3]
                    hand_features = line[4]
                elif set_type == 'test':
                    question_a = line[2]
                    question_b = line[3]
                    category = line[0]
                    hand_features = line[4]
                    label = 0
                else:
                    raise ValueError()
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, question_a=question_a, question_b=question_b, label=label, category=category, hand_features=hand_features))
        return examples


def convert_examples_to_features(examples, label_list, category_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token='[CLS]', cls_token_segment_id=1,
                                 sep_token='[SEP]', sep_token_extra=False,
                                 pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_toekn_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    category_map = {category: i for i, category in enumerate(category_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        hand_features = example.hand_features

        if ex_index % 10000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.question_a)
        tokens_b = tokenizer.tokenize(example.question_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with '- 3'. '- 4' for ro RoBERTa
        special_tokens_count = 4  if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-special_tokens_count)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        category_clf_tokens = tokens_a + tokens_b
        special_tokens_count = 3 if sep_token_extra else 2
        if len(category_clf_tokens) > max_seq_length - special_tokens_count:
            category_clf_tokens = category_clf_tokens[:(max_seq_length - special_tokens_count)]
        category_clf_tokens += [sep_token]
        category_clf_segment_ids = [sequence_a_segment_id] * len(category_clf_tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

            category_clf_tokens = [cls_token] + category_clf_tokens
            category_clf_segment_ids = [cls_token_segment_id] + category_clf_segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        category_clf_input_ids = tokenizer.convert_tokens_to_ids(category_clf_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        category_clf_input_mask = [1 if mask_padding_with_zero else 0] * len(category_clf_input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        category_clf_padding_length = max_seq_length - len(category_clf_input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

            category_clf_input_ids = ([pad_token] * category_clf_padding_length) + category_clf_input_ids
            category_clf_input_mask = ([0 if mask_padding_with_zero else 1] * category_clf_padding_length) + category_clf_input_mask
            category_clf_segment_ids = ([pad_token_segment_id] * category_clf_padding_length) + category_clf_segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            category_clf_input_ids = category_clf_input_ids + ([pad_token] * category_clf_padding_length)
            category_clf_input_mask =  category_clf_input_mask + ([0 if mask_padding_with_zero else 1] * category_clf_padding_length)
            category_clf_segment_ids = category_clf_segment_ids + ([pad_token_segment_id] * category_clf_padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(category_clf_input_ids) == max_seq_length
        assert len(category_clf_input_mask) == max_seq_length
        assert len(category_clf_segment_ids) == max_seq_length

        label_id = label_map[example.label]
        category_id = category_map[example.category]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("hand_features: %s" % " ".join([str(x) for x in hand_features]))
            if example.label is not None:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          category_clf_input_ids=category_clf_input_ids,
                          category_clf_input_mask=category_clf_input_mask,
                          category_clf_segment_ids=category_clf_segment_ids,
                          category_id=category_id,
                          hand_features=hand_features
                          ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """ Truncates a sequence pair in place to the maximum length. """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information that a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
    return (preds == labels).mean()
