# -*- coding: utf-8 -*-
""" Extract the pretrained character level embedding from BERT hidden outputs. """
import re

import numpy as np
from pytorch_transformers import BertTokenizer, BertModel


if __name__ == '__main__':
    print('# Load pretrained model tokenizer.')
    tokenizer = BertTokenizer.from_pretrained('./bert_wwm/')

    print('# Construct vocab.')
    vocab = [token for token in tokenizer.vocab]

    print('# Load pretrained model.')
    model = BertModel.from_pretrained('./bert_wwm')

    print('# Load word embeddings')
    emb = model.embeddings.word_embeddings.weight.data
    emb = emb.numpy()

    print('# Write')
    with open('{}.{}.{}d.vec'.format('bert_wwm', len(vocab), emb.shape[-1]), 'w', encoding='utf-8') as fout:
        fout.write('{} {}\n'.format(len(vocab), emb.shape[-1]))
        assert len(vocab) == len(emb), 'The number of vocab and embeddings MUST be identical.'
        for token, e in zip(vocab, emb):
            e = np.array2string(e, max_line_width=np.inf)[1:-1]
            e = re.sub('[ ]+', ' ', e)
            fout.write('{} {}\n'.format(token, e))
