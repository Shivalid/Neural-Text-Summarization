#!/usr/bin/env python
# coding: utf-8

# In[25]:


os.chdir('C:/WS 2019/Neural Text Summarization/project')
path = os.getcwd()


# In[29]:


from extractor import PacSumExtractorWithBert, PacSumExtractorWithTfIdf
from data_iterator import Dataset

import argparse


extractor = PacSumExtractorWithBert(bert_model_file = 'C:/WS 2019/Neural Text Summarization/project/pacssum_models/pytorch_model_finetuned.bin',
                                            bert_config_file = 'C:/WS 2019/Neural Text Summarization/project/pacssum_models/bert_config.json',
                                            beta = 0.0,
                                            lambda1=0.0,
                                            lambda2=1.0)
        


# In[ ]:


#tune
        if args.mode == 'tune':
            tune_dataset = Dataset(args.tune_data_file, vocab_file = args.bert_vocab_file)
            tune_dataset_iterator = tune_dataset.iterate_once_doc_bert()
            extractor.tune_hparams(tune_dataset_iterator)

        #test
        test_dataset = Dataset(args.test_data_file, vocab_file = args.bert_vocab_file)
        test_dataset_iterator = test_dataset.iterate_once_doc_bert()
        extractor.extract_summary(test_dataset_iterator)

