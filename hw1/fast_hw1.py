import numpy as np
from scipy.sparse import csr_matrix

import pandas as pd
import re
import xml.etree.ElementTree as ET

import argparse
from time import time
from functools import reduce

import os
from os.path import join

import numpy as np
from scipy.sparse import csr_matrix

import pandas as pd
import re
import xml.etree.ElementTree as ET

from time import time
from functools import reduce

import os
from os.path import join

# ----- Evaluation -----

def AveP(pred, ans):
    is_relevence = np.array([1 if p in ans else 0 for p in pred])
    return ((np.arange(sum(is_relevence))+1) / (np.where(is_relevence > 0)[0]+1)).sum() / min(100, len(ans))

# ----- Query Processing -----

def read_query(filepath):
    tree = ET.parse(filepath)
    questions = []
    for p in tree.findall('topic'):
        questions.append([x.text.strip() for x in p.findall('*')])
    return questions

def query_processing(questions):
    def parse(words):
        uni = np.array(reduce(np.union1d, [list(x) for x in words]))
        bi = np.array(reduce(np.union1d, [word[i:i+2] for word in words for i in range(len(word)-1)]))
        return uni.tolist(), bi.tolist()
    
    res = []
    for q in questions:
        concepts = parse(q[-1][:-1].split('、'))
        title = parse([q[1]])
        res.append((concepts, title))
    return res

def query_extractor(d, terms, return_filter=True):
    uni, bi = terms
    bi = np.setdiff1d(bi, d.black_list).tolist()
    query_terms = [d.vocabs_dict[x] for x in uni+bi]
    if return_filter:
        return query_terms, np.ones(len(query_terms)), bi
    return query_terms, np.ones(len(query_terms))

# ----- Format handler -----

def open_file(path):
    tree = ET.parse(path.strip())
    id_ = tree.find('.//id').text
    text = ''.join([x.text.strip() for x in tree.findall('.//p')])
    return id_, text

def get_result_list(q_id, filelist, res):
    return [q_id[-3:], ' '.join([open_file(filelist[i])[0].lower() for i in res])]

# ----- Search engine -----

def BM25_score(d, weight, terms, cand, k1=1.2, b=0.75):
    tf = d.t2d[terms, :][:, cand].toarray()
    return (weight*d.idf[terms]).dot((tf*(k1 + 1))/(tf + k1*(1 - b + b*(d.docs_length[cand] / d.avg_length))))

def search(scores, cand, num=100):
    rank = np.argsort(scores)
    res = cand[rank[-num:]][::-1]
    return rank, res

# ----- Dataset -----

class Dataset:
    def __init__(self, model_path, ntcir_path):        
        self.filelist = []
        
        self.docs_length = np.zeros(0)
        self.avg_length = 0
        self.num_docs = 0
        
        self.vocabs_dict = {}
        self.black_list = []
        self.idf = np.zeros(0)
        self.t2d = None
        
        self.start = 0
        self.model_path = model_path
        self.ntcir_path = ntcir_path
        
    def get_docs_length(self):
        with open(join(self.model_path, 'file-list')) as f:
            self.filelist = [join(self.ntcir_path, s.strip()) for s in f.readlines()]
            for file_path in self.filelist:
                tree = ET.parse(file_path)
                text = ''.join([x.text.strip() for x in tree.findall('.//p')])
                chinese_text = re.findall(r"[\u4e00-\u9fa5']+", text)
                self.docs_length = np.r_[self.docs_length, np.sum([len(x) for x in chinese_text])]
        self.avg_length = np.mean(self.docs_length)
        self.num_docs = len(self.docs_length)
    
    def dump_time(self, slogan):
        print(slogan+', total time: %06.2f sec.' % (time() - self.start))

    def build(self, corpus):
        self.start = time()
        self.get_docs_length()
        self.dump_time('Finish getting documents length')
        
        # Read inverted-file
        all_term = pd.read_csv(join(self.model_path, 'inverted-file'), delimiter=' ', header=None, usecols=[0,1,2]).values
        # Get the indices of lines with 3 digits
        indices = np.where(~np.isnan(all_term[:, 2]))[0]
        
        self.dump_time('Finish reading inverted file')
        
        # Read vocab.all
        char = pd.read_csv(join(self.model_path, 'vocab.all'), header=None, index_col=False, delimiter='\n', quoting=3, encoding='utf-8').values.reshape(-1)
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
        self.dump_time('\nFinish building dataset')

# ----- Arguments Parser ----- 

def arguments_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--relevence', required=False, action='store_true', default=False,
            help='Turn on the relevance feedback in the program')
    parser.add_argument('-a', required=False, dest='answer_path',
            help='The answer file (if exist)')
    parser.add_argument('-i', required=True, dest='query_file', 
            help='The input query file')
    parser.add_argument('-o', required=True, dest='output_file', 
            help='The output ranked list file')
    parser.add_argument('-m', required=True, dest='model_path', 
            help='The input model directory')
    parser.add_argument('-d', required=True, dest='nctir_path', 
            help='The directory of NTCIR documents')
    return parser.parse_args()

# ----- Main function -----

if __name__ == '__main__':
    start = time()
    args = arguments_parsing()
    
    if args.answer_path:
        answer = [x.split() for x in pd.read_csv(args.answer_path)['retrieved_docs'].tolist()]
        
    questions = read_query(args.query_file)
    queries = query_processing(questions)
    corpus = reduce(np.union1d, [x[0]+x[1]+y[0]+y[1] for x, y in queries]).tolist()

    d = Dataset(args.model_path, args.nctir_path)
    d.build(corpus)
    
    result = []
    if args.answer_path:
        MAP, MRP = [], []

    # Parameters of relevence feedback
    num_iter = 1 if not args.relevence else 5
    num_related, alpha = 5, 0.98

    if args.relevence:
        print('Start Relevence Feedback Mode, num_iter: %d, num_related: %d, alpha: %.2f' %(num_iter, num_related, alpha))

    for _, query in enumerate(queries):
        print('Processing %02d / %02d, total time: %06.2f sec.' % (_+1, len(queries), time() - start), end='\r')

        concepts, title = query
        query_terms, query_weight, filters = query_extractor(d, concepts, return_filter=True)
        candidates = reduce(np.union1d, [d.t2d[d.vocabs_dict[x]].nonzero() for x in filters])

        for __ in range(num_iter):
            cscores = BM25_score(d, query_weight, query_terms, candidates)
            crank, cres = search(cscores, candidates)
            related_weight = d.t2d[query_terms, :][:, cres[:num_related]].toarray().mean(axis=-1)
            query_weight = alpha * query_weight + (1 - alpha) * related_weight

        # Using title to further improve accuracy
        t_query_terms, t_query_weight = query_extractor(d, title, return_filter=False)
        candidates = candidates[crank[-2000:]]
        tscores = BM25_score(d, t_query_weight, t_query_terms, candidates)

        # Mix the results from concepts, and from title
        beta = 0.5
        scores = beta * cscores[crank[-2000:]] + (1-beta) * tscores
        rank, res = search(scores, candidates)

        result.append(get_result_list(questions[_][0], d.filelist, res))

        if args.answer_path:
            MAP.append(AveP(result[-1][1].split(), answer[_]))
            recall = np.sum([1 if r in answer[_] else 0 for r in result[-1][1].split()[:30]])
            MRP.append(recall / len(answer[_]))

    pd.DataFrame(result).to_csv(args.output_file, header=['query_id','retrieved_docs'], index=False)
    print('\nFinish, total time: %06.2f sec.' % (time() - start))

    if args.answer_path:
        print('MAP@100: %.5f' % (np.mean(MAP)))
        print('MRP: %.5f' % (np.mean(MRP)))
