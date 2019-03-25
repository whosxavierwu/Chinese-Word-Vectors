# -*- coding:utf-8 -*-
# Created by: wuzewei
# Created on: 2019/3/25 0025

from gensim.models import KeyedVectors
import numpy as np
from pyhanlp import HanLP


class WordVectorFetcher:
    def __init__(self, filename):
        self.wv_filename = filename
        self.wv = None

    def init(self):
        self.wv = KeyedVectors.load_word2vec_format(self.wv_filename)

    def get_word_vector(self, word):
        if word not in self.wv:
            return np.zeros(self.wv.vector_size)
        else:
            return self.wv[word]

    def get_sentence_vector(self, sentence):
        words = [item.word for item in HanLP.segment(sentence)]
        cnt = 0
        vec_fin = np.zeros(self.wv.vector_size)
        for w in words:
            if w in self.wv:
                vec_fin += self.get_word_vector(w)
                cnt += 1
        vec_fin = vec_fin / cnt
        return vec_fin


if __name__ == '__main__':
    fn = 'SGNS/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/tmp.txt'
    fetcher = WordVectorFetcher(fn)
    fetcher.init()
    print(fetcher.get_sentence_vector(u'在的了'))

