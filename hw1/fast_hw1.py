import numpy as np
from scipy.sparse import csr_matrix

import pandas as pd
import re
import xml.etree.ElementTree as ET

from time import time
from functools import reduce

import os
from os.path import join

def read_query(filepath):
    tree = ET.parse(filepath)
    questions = []
    for p in tree.findall('topic'):
        questions.append([x.text.strip() for x in p.findall('*')])
    return questions

def query_processing(questions):
    questions = [x[-1][:-1].split('、') for x in questions]
    res = []
    for q in questions:
        uni_qspace = set()
        bi_qspace = set()
        for words in q:
            uni_qspace |= set(list(words))
            bi_qspace |= set([words[i:i+2] for i in range(len(words)-1)])
        res.append((list(uni_qspace), list(bi_qspace)))
    return res

def open_file(path):
    tree = ET.parse(path.strip())
    id_ = tree.find('.//id').text
    text = ''.join([x.text.strip() for x in tree.findall('.//p')])
    return id_, text
    
def BM25_score(d, terms, cand, k1=1.2, b=0.75):
    tf = d.t2d[terms, :][:, cand].toarray()
    return d.idf[terms].dot((tf*(k1 + 1))/(tf + k1*(1 - b + b*(d.docs_length[cand] / d.avg_length))))

def get_result_list(q_id, filelist, res):
    return [q_id, ' '.join([open_file(filelist[i])[0].lower() for i in res])]

class Dataset:
    def __init__(self):        
        self.filelist = []
        
        self.docs_length = np.zeros(0)
        self.avg_length = 0
        self.num_docs = 0
        
        self.vocabs_dict = {}
        self.black_list = []
        self.idf = np.zeros(0)
        self.t2d = None
        
        self.start = 0
        
    def get_docs_length(self, path):
        with open(join(path, 'model/file-list')) as f:
            self.filelist = [join(path, 'CIRB010', s.strip()) for s in f.readlines()]
            for file_path in self.filelist:
                tree = ET.parse(file_path)
                text = ''.join([x.text.strip() for x in tree.findall('.//p')])
                chinese_text = re.findall(r"[\u4e00-\u9fa5']+", text)
                self.docs_length = np.r_[self.docs_length, np.sum([len(x) for x in chinese_text])]
        self.avg_length = np.mean(self.docs_length)
        self.num_docs = len(self.docs_length)
    
    def dump_time(self, slogan):
        print(slogan+', total time: %06.2f sec.' % (time() - self.start))
    def build(self, corpus, path):
        self.start = time()
        self.get_docs_length(path)
        self.dump_time('Finish getting documents length')
        
        # Read inverted-file
        all_term = pd.read_csv(join(path, 'model/inverted-file'), delimiter=' ', header=None, usecols=[0,1,2]).values
        # Get the indices of lines with 3 digits
        indices = np.where(~np.isnan(all_term[:, 2]))[0]
        
        self.dump_time('Finish reading inverted file')
        
        # Read vocab.all
        char = pd.read_csv(join(path, 'model/vocab.all'), header=None, index_col=False, delimiter='\n', quoting=3, encoding='utf-8').values.reshape(-1)
        char_dict = dict(zip(char, np.arange(len(char), dtype=int)))
        
        self.dump_time('Finish reading vocabulary file')
        
        terms = [(char_dict[x[0]], char_dict[x[1]]) if len(x) > 1 else (char_dict[x[0]], -1) for x in corpus]
        terms = sorted(terms, key=lambda t: (t[0], t[1]))
        i = 0
        row, col, data = [], [], []
        for t1, t2 in terms:
            w = char[t1] + char[t2] if t2 > 0 else char[t1]
            for move, idx in enumerate(indices[i:]):
                # Find the term
                if all_term[idx][0] == t1 and all_term[idx][1] == t2:
                    # Add the term to vocabs_dict
                    self.vocabs_dict[w] = len(self.vocabs_dict)

                    nqi = all_term[idx][2]
                    self.idf = np.r_[self.idf, np.log((self.num_docs - nqi + 0.5)/(nqi + 0.5) + 1)]
                    interval = all_term[idx+1:indices[i+move+1]].astype(int)

                    row += [len(self.vocabs_dict)-1]*len(interval)
                    col += interval[:, 0].tolist()
                    data += interval[:, 1].tolist()
                    break
                    
                # The term doesn't exist in inverted-file
                elif all_term[idx][0] > t1:
                    self.black_list.append(w)
                    break
            i += move
            print('Processing ... %06.2f%%, total time: %06.2f sec.' % (100*(i+1)/len(indices), time() - self.start), end='\r')
        self.t2d = csr_matrix((data, (row, col)), shape=(len(self.idf), self.num_docs))
        self.idf = np.array(self.idf)
        self.dump_time('\nFinish building dataset')

if __name__ == '__main__':
    start = time()
    data_path = '/tmp2/r09922104/ir'
    train_path = join(data_path, 'queries/query-train.xml')
    test_path = join(data_path, 'queries/query-test.xml')

    questions = read_query(train_path) + read_query(test_path)
    queries = query_processing(questions)
    corpus = reduce(np.union1d, [x+y for x, y in queries]).tolist()

    d = Dataset()
    d.build(corpus, data_path)

    result = []
    for _, query in enumerate(queries):
        print('Processing %02d / %02d, total time: %06.2f sec.' % (_+1, len(queries), time() - start), end='\r')
        uni, bi = query
        bi = np.setdiff1d(bi, d.black_list).tolist()
        candidates = reduce(np.union1d, [d.t2d[d.vocabs_dict[x]].nonzero() for x in bi])
        
        query_terms = [d.vocabs_dict[x] for x in uni+bi]
        
        scores = BM25_score(d, query_terms, candidates)
        rank = np.argsort(scores)
        res = [candidates[i] for i in rank[-100:][::-1]]
        result.append(get_result_list('%03d' % (_+1), d.filelist, res))
    pd.DataFrame(result).to_csv('out.csv', header=['query_id','retrieved_docs'], index=False)
    print('\nFinish, total time: %06.2f sec.' % (time() - start))
