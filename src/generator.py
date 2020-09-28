'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-27 15:10:06
 * @desc 
'''
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from random import choice
from utils import transform_subs,transform_objs ,transform_Csubs
RANDOM_SEED = 2020

class HBTDataSet(object):
    def __init__(self,file_path,tokenizer,rel_list,batch_size):
        with open(file_path , 'r' , encoding = 'utf-8') as f:
            self.data=json.load(f)
        
        self.tokenizer = tokenizer
        
        self.rel2id = { rel:i for i,rel in enumerate(rel_list)}
        self.num_rels = len(self.rel2id)

        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size

        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    
    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idxs)
            Text, Subs, Csub, RO = [], [], [], []
            for i,idx in enumerate(idxs):
                item = self.data[i]
                text = item['text']
                triple_list = item['triple_list']
                d = {}
                for triple in triple_list:
                    sub,rel,obj = triple
                    if sub in d:
                        d[sub].append((self.rel2id[rel],obj))
                    else:
                        d[sub] = [(self.rel2id[rel],obj)]
                subs = list(d.keys())
                choice_sub = choice(subs)
                triple = d[choice_sub]

                Text.append(text)
                Subs.append(subs)
                Csub.append(choice_sub)
                RO.append(triple)

                if (i+1) % self.batch_size == 0 or i+1 == len(self.data):
                    input_ids , _ ,attention_mask = self.tokenizer(Text,padding=True).values()
                    tokenize = lambda x: self.tokenizer.encode(x,add_special_tokens = False)
                    Sub_start,Sub_end = transform_subs(input_ids, attention_mask, Subs, tokenize)
                    sub_pos = transform_Csubs(input_ids, attention_mask, Csub, tokenize)
                    Obj_start,Obj_end = transform_objs(input_ids, attention_mask, RO, tokenize,self.num_rels)
                    
                    F = lambda x : torch.Tensor(x).long()
                    input_ids = F(input_ids)
                    attention_mask = F(attention_mask)
                    sub_pos = F(sub_pos)

                    F = lambda x: torch.Tensor(x).float()
                    
                    yield input_ids, attention_mask, F(Sub_start), F(Sub_end), sub_pos, F(Obj_start), F(Obj_end)
                    Text, Subs, Csub, RO = [],[],[],[]

if __name__ == '__main__':    
    '''
    Unit Test
    '''

    from transformers import BertTokenizer
    import os
    cwd = os.getcwd()
    tokenizer_path = os.path.join(cwd, 'data', 'static', 'bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    dev_path = os.path.join(cwd, 'data', 'WebNLG', 'dev.json')
    rel_path = os.path.join(cwd, 'data', 'WebNLG', 'relation_type.txt')
    with open(rel_path, 'r', encoding = 'utf-8') as f:
        rel_list =[ line.strip() for line in  f.readlines() ]
    
    data_loader =  HBTDataSet(dev_path, tokenizer, rel_list, 32)



    for data in data_loader:
        import pdb; pdb.set_trace()
        x,y,z,a,b,c,d = data

    
    