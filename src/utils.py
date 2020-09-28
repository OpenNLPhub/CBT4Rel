'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-27 15:50:04
 * @desc 
'''
import numpy as np
import re
from collections import Counter
from sklearn.metrics import confusion_matrix
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
        # spos[start] = 1.
        # spos[end] = 1.
        for i in range(start,end+1):
            spos[i] = 1.
        Map.append(spos)

    return Map


def transform_objs(input_ids, attention_mask, RO ,tokenize,rel_nums):
    max_len = len(input_ids[0])
    S,E = [],[]
    # import pdb
    # pdb.set_trace()
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
            
    

        
        
def focal_loss(y_true,y_pred,device):
    alpha,gamma = torch.tensor(0.25).to(device) , torch.tensor(2.0).to(device)
    y_pred=torch.clamp(y_pred,1e-7,1-1e-7)
    return - alpha * y_true * torch.log(y_pred) * (1 - y_pred) ** gamma\
        - (1 - alpha) * (1 - y_true) * torch.log(1 -  y_pred) * y_pred


def binary_confusion_matrix_evaluate(y_true,y_pred):
    # import pdb; pdb.set_trace()
    tn ,fp, fn, tp =  confusion_matrix(y_true,y_pred).ravel()
    acc = float(tn + tp)/(fp+fn+tn+tp)
    prec =  float(tp) / (tp + fp) if (tp+fp) != 0 else 0.
    recall =  float(tp) / (tp + fn) if (tp + fn) != 0 else 0.
    f1= 2*prec*recall / ( prec + recall) if prec + recall !=0 else 0.
    return acc,prec,recall,f1

def confusion_matrix(y_true,y_pred):
    return confusion_matrix(y_true,y_pred).ravel()

def evaluate(x):
    tn , fp , fn , tp = x
    acc = float(tn + tp)/(fp+fn+tn+tp)
    prec =  float(tp) / (tp + fp) if (tp+fp) != 0 else 0.
    recall =  float(tp) / (tp + fn) if (tp + fn) != 0 else 0.
    f1= 2*prec*recall / ( prec + recall) if prec + recall !=0 else 0.
    return acc,prec,recall,f1