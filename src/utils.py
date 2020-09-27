'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-27 15:50:04
 * @desc 
'''
import numpy as np
import re
from collections import Counter
''' ---------------- Generate Dataset ---------------- '''
def find_index(a,b):
    l = len(b)
    head = b[0]
    ans = []
    for i,v in enumerate(a):
        if v == head and a[i:i+l] == b:
            ans.append((i,i+l-1))
    return ans


def transform_subs(input_ids,attention_mask,Subs,tokenize):
    max_len = len(input_ids[0])
    S,E = [],[]
    for i,subs in enumerate(Subs):
        l = np.sum(attention_mask[i])
        text = input_ids[i][:l]
        _subs = [ tokenize(sub) for sub in subs]

        sub_start = np.zeros(max_len)
        sub_end = np.zeros(max_len)

        for sub in _subs:
            start_end=find_index(text,sub)
            for start,end in start_end:
                sub_start[start] = 1.
                sub_end[end] = 1.
            
        S.append(sub_start)
        E.append(sub_end)
    return S,E   

def transform_Csubs(input_ids, attention_mask, Csub, tokenize):
    max_len = len(input_ids[0])
    Map = []
    for i,sub in enumerate(Csub):
        l = np.sum(attention_mask[i])
        text = input_ids[i][:l]
        spos = np.zeros(max_len)
        _sub = tokenize(sub)
        start, end = find_index(text, _sub)[0]
        spos[start] = 1.
        spos[start] = 1.
        Map.append(spos)

    return spos


def transform_objs(input_ids, attention_mask, RO ,tokenize,rel_nums):
    max_len = len(input_ids[0])
    S,E = [],[]
    import pdb
    pdb.set_trace()
    for i,triples in enumerate(RO):
        l = np.sum(attention_mask[i])
        text = input_ids[i][:l]
        obj_start = np.zeros((max_len,rel_nums))
        obj_end = np.zeros((max_len,rel_nums))
        for v in triples:
            relid,obj = v
            _obj = tokenize(obj)
            start, end = find_index(text, _obj)[0]
            obj_start[start][relid] = 1.
            obj_end[end][relid] = 1.
        S.append(obj_start)
        E.append(obj_end)
    return S,E
            
    

        
        
