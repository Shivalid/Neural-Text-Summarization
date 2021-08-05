#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from collections import Counter
import numpy as np
import math
import torch
import torch.nn as nn
import random
import time
import io
import codecs

import rougee
from extractor import PacSumExtractorWithBert, PacSumExtractorWithTfIdf
from data_iterator import Dataset
import rougee


# In[ ]:


tune_dataset = Dataset('C:/WS 2019/Neural Text Summarization/project/data/CNN_DM/cd.validation.h5df')
tune_dataset_iterator = tune_dataset.iterate_once_doc_bert()


# In[ ]:


extractor = PacSumExtractorWithTBert()


# In[ ]:


def _select_tops(edge_scores, beta, lambda1, lambda2):

        min_score = edge_scores.min()
        max_score = edge_scores.max()
        edge_threshold = min_score + beta * (max_score - min_score)
        new_edge_scores = edge_scores - edge_threshold
        forward_scores, backward_scores, _ = _compute_scores(new_edge_scores, 0)
        forward_scores = 0 - forward_scores

        paired_scores = []
        for node in range(len(forward_scores)):
            paired_scores.append([node,  lambda1 * forward_scores[node] + lambda2 * backward_scores[node]])

        #shuffle to avoid any possible bias
        random.shuffle(paired_scores)
        paired_scores.sort(key = lambda x: x[1], reverse = True)
        extracted = [item[0] for item in paired_scores[:3]]


        return extracted

def _compute_scores(similarity_matrix, edge_threshold):

    forward_scores = [0 for i in range(len(similarity_matrix))]
    backward_scores = [0 for i in range(len(similarity_matrix))]
    edges = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix[i])):
            edge_score = similarity_matrix[i][j]
            if edge_score > edge_threshold:
                forward_scores[j] += edge_score
                backward_scores[i] += edge_score
                edges.append((i,j,edge_score))

    return np.asarray(forward_scores), np.asarray(backward_scores), edges


# In[ ]:


def _tune_extractor(edge_scores):

        tops_list = []
        hparam_list = []
        num = 10
        for k in range(num + 1):
            beta = k / num
            for i in range(11):
                lambda1 = i/10
                lambda2 = 1 - lambda1
                extracted = _select_tops(edge_scores, beta=beta, lambda1=lambda1, lambda2=lambda2)

                tops_list.append(extracted)
                hparam_list.append((beta, lambda1, lambda2))

        return tops_list, hparam_list


# In[ ]:


example_num=1000


summaries, references = [], []
k = 0
for item in tune_dataset_iterator:
    
    article, abstract, inputs = item
    edge_scores =extractor._calculate_similarity_matrix(*inputs)
    tops_list, hparam_list = _tune_extractor(edge_scores)

    summary_list = [list(map(lambda x: article[x], ids)) for ids in tops_list]
    summaries.append(summary_list)
    references.append([abstract])
    k += 1
    print(k)
    if k % example_num == 0:
        break


# In[ ]:


count = 0
best_rouge = 0
best_hparam = None
for i in range(len(summaries[0])):
    print(i)
    print("threshold :  "+str(hparam_list[i])+'\n')
    #print("non-lead ratio : "+str(ratios[i])+'\n')
    for k in range(len(summaries)):
        #print(i)
        print(k)
        summ = summaries[k][i]
        #print('summ')
        #print(len(summ))
        #print(summaries)
        ref = references[k][0]
        
        if len(summ) == len(ref):
            count = count + 1
            
            print('sum')
            print(len(summ))
            if i != 8:
                if summaries[k][i][0]!='.' and summaries[k][i][1]!='.' and summaries[k][i][2]!='.':
                    result = rouge.get_scores(summ, ref, avg = True)
                    #print(result)
                    if result['rouge-l']['f'] > best_rouge:
                        best_rouge = result['rouge-l']['f']
                        best_hparam = hparam_list[i]

print("The best hyper-parameter :  beta %.4f , lambda1 %.4f, lambda2 %.4f " % (best_hparam[0], best_hparam[1], best_hparam[2]))
print("The best rouge_1_f_score :  %.4f " % best_rouge)

beta = best_hparam[0]
lambda1 = best_hparam[1]
lambda2 = best_hparam[2]

