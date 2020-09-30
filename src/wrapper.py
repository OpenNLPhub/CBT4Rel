


import torch
import numpy as np
from transformers import BertModel
import config
import torch.optim as optim
from model import CBT
from utils import focal_loss
from log import logger
from copy import deepcopy
from utils import confusion_matrix,evaluate,find_index
import json


class wrapper(object):

    def __init__(self,rel_list):
        self.rel_list = rel_list

        self.device = torch.device( 'cuda:1' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel.from_pretrained(config.BertPath)
        
        self.model =  CBT(self.bert,len(self.rel_list)).to(self.device)
        self.epoches = 100
        self.lr = 1e-5

        self.best_model = CBT(self.bert,len(self.rel_list)).to(self.device)
        self.best_loss = 1e12
        self.print_step = 15
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)

        self.loss = lambda y_true,y_pred : focal_loss (y_true,y_pred,self.device)

    def _cal_loss(self, x ,x_ ,attention_mask):
        pred_sub_start, pred_sub_end , pred_obj_start , pred_obj_end = x_
        Sub_start, Sub_end, Obj_start, Obj_end = x
        
        sub_start_loss = torch.sum(self.loss(Sub_start,pred_sub_start) * attention_mask) / torch.sum(attention_mask)
        sub_end_loss = torch.sum(self.loss(Sub_end,pred_sub_end) * attention_mask) / torch.sum(attention_mask)

        obj_start_loss = torch.sum(self.loss(Obj_start,pred_obj_start), axis = -1)
        obj_start_loss = torch.sum( obj_start_loss * attention_mask) / torch.sum(attention_mask)

        obj_end_loss = torch.sum(self.loss(Obj_end,pred_obj_end), axis = -1)
        obj_end_loss = torch.sum(obj_end_loss * attention_mask) / torch.sum(attention_mask)

        return sub_start_loss + sub_end_loss + obj_start_loss + obj_end_loss

    def _decode(self,x,x_,attention_mask,threshold = 0.5):
        mask = attention_mask == 1
        y = x.masked_select(mask).cpu().long().numpy()
        y_ = x_.masked_select(mask).cpu().numpy()
        y_ = np.where(y_ > threshold ,1 ,0)
        # np.array vector float
        return confusion_matrix(y,y_)
        # tn,fp,fn,tp
    
    def _metric(self,x,x_,attention_mask):
        d = {}
        Sub_start, Sub_end, Obj_start, Obj_end = x
        pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end = x_
        d['sub_start'] = self._decode(Sub_start, pred_sub_start,attention_mask)
        d['sub_end'] = self._decode(Sub_end, pred_sub_end,attention_mask)

        #Obj batch_size * max_seq_len * rel_num
        attention_mask_ = attention_mask.unsqueeze(-1).expand(-1,-1,pred_obj_start.shape[-1])
        d['obj_start'] = self._decode(Obj_start, pred_obj_start,attention_mask_)
        d['obj_end'] = self._decode(Obj_end, pred_obj_end,attention_mask_)
        return d

        

    def train(self,train_dataloader, dev_dataloader): 
        for epoch in range(1,self.epoches+1):
            self.model.train()
            # self.optimizer.
            for ix,item in enumerate(train_dataloader):
                

                input_ids, attention_mask, Sub_start, Sub_end, sub_pos\
                    ,Obj_start, Obj_end = [i.to(self.device) for i in item]
                
                pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end \
                    = self.model([input_ids,attention_mask,sub_pos])
                
                x = [Sub_start, Sub_end, Obj_start, Obj_end]
                x_ = [pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end]
                loss = self._cal_loss(x,x_,attention_mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (ix+1) % self.print_step == 0 or ix == len(train_dataloader) - 1:
                    logger.info("[In Training] Epoch : {} \t Step/All_Step : {}/{} \t Loss of every word : {}"\
                        .format(epoch,(ix+1),len(train_dataloader),loss.item()))
            
            #validation
            self.model.eval()
            with torch.no_grad():
                d = {
                    'obj_start' : [],
                    'obj_end' : [],
                    'sub_start' : [],
                    'sub_end' : []
                }
                val_loss = []
                
                for ix,item in enumerate(dev_dataloader):
                    input_ids, attention_mask, Sub_start, Sub_end\
                        ,sub_pos, Obj_start, Obj_end = [i.to(self.device) for i in item]
                    
                    pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end\
                        = self.model([input_ids, attention_mask, sub_pos])
            
                    x = [Sub_start, Sub_end, Obj_start, Obj_end]
                    x_ = [pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end]
                    loss = self._cal_loss(x, x_, attention_mask)
                    val_loss.append(loss.item())

                    if (ix+1) % (self.print_step) == 0 or ix == len(dev_dataloader) - 1:
                        logger.info("[In Validation] Step/All_Step : {}/{} \t Loss of every word : {}"\
                            .format((ix+1),len(dev_dataloader),loss.item()))
                    
                    #decode
                    dd = self._metric(x,x_,attention_mask)
                    for key in d.keys():
                        d[key].append(dd[key])

            eval_loss = sum(val_loss) / len(val_loss)
            logger.info("In Validation : The Average Loss is {}".format(eval_loss))
            if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.best_model = deepcopy(self.model)
                    logger.info("Best Model UpDate !")

            for key,value in d.items():
                m =  list(zip(*value))
                # m = [tn,fp,fn,tp]
                m = [ sum(i) for i in m]
                acc, prec, recall, f1 = evaluate(m)
                logger.info("Epoch {} Validation: Label:{}\t Acc : {}\t Precision:{}\t Recall:{}\t f1-score:{}\t"\
                    .format(epoch,key,acc,prec,recall,f1))
            
            torch.cuda.empty_cache()

    def test(self,test_dataloader):
        self.best_model.eval()
        with torch.no_grad():
            d = {
                    'obj_start' : [],
                    'obj_end' : [],
                    'sub_start' : [],
                    'sub_end' : []
                }
            for ix,item in enumerate(test_dataloader):
                input_ids, attention_mask, Sub_start, Sub_end\
                    ,sub_pos, Obj_start, Obj_end = [i.to(self.device) for i in item]
                
                pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end\
                    = self.best_model([input_ids, attention_mask, sub_pos])
        
                x = [Sub_start, Sub_end, Obj_start, Obj_end]
                x_ = [pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end]
                loss = self._cal_loss(x, x_, attention_mask)

                #decode
                dd = self._metric(x,x_,attention_mask)
                for key in d.keys():
                    d[key].append(dd[key])
         
        for key,value in d.items():
            m =  list(zip(*value))
            # m = [tn,fp,fn,tp]
            m = [ sum(i) for i in m]
            acc, prec, recall, f1 = evaluate(m)
            logger.info("Test: Label:{}\t Acc : {}\t Precision:{}\t Recall:{}\t f1-score:{}\t"\
                .format(key,acc,prec,recall,f1))

    def load_model(self,PATH):
        self.model.load_state_dict(torch.load(PATH))
        self.best_model.load_state_dict(torch.load(PATH))
        self.best_model.eval()
        logger.info('successfully load model')

    def save_model(self,PATH):
        torch.save(self.best_model.state_dict(),PATH)
        logger.info('save best model successfully')

    
    def extract_triple(self, data, tokenizer, filepath, threshold=0.5):
        '''
        data item{
            "text":" ... ",
            "triple_list": [[s,r,o],[s,r,o],[s,r,o],[s,r,o]]
        }
        '''
        self.best_model.eval()
        id2rel= { i:v for i,v  in enumerate(self.rel_list)}
        tp = 0
        a = 0
        b = 0
        tostr =  lambda x : ' '.join([ str(i) for i in x])
        with torch.no_grad():
            for item in data:
                text = item['text']
                triple_lists = item['triple_list']
                #process triple to string 
                # convenient to equal

                # y end2end test 
                y = {}
                y_ = []
                ans_triple = []

                for triple in triple_lists:
                    s,r,o = triple
                    s = tokenizer.encode(s, add_special_tokens = False)
                    o = tokenizer.encode(o, add_special_tokens = False)
                    r = r
                    t = " ".join([tostr(s),r,tostr(o)])
                    y[t] = 1

                input_ids_ , _ ,attention_mask_ =tokenizer(text, padding = True).values()
                #input_ids:  max_seq_len
                input_ids = torch.Tensor(input_ids_).long().unsqueeze(0)
                attention_mask = torch.Tensor(attention_mask_).long().unsqueeze(0)
                #input_ids: 1 * max_seq_len

                x = [ i.to(self.device) for i in [input_ids,attention_mask]]
                pred_sub_start, pred_sub_end = self.best_model.subject_tag(x)
                # 1 * max_seq_len
                F = lambda x : x.cpu().squeeze(0).numpy()

                pred_sub_start = np.where(F(pred_sub_start) > threshold, 1, 0)
                pred_sub_end = np.where(F(pred_sub_end) > threshold, 1, 0)
                # max_seq_len           
                subject_pos = self._extract_entity(pred_sub_start, pred_sub_end)
                
                subject_token = [ tostr(input_ids_[i[0]:i[1]+1]) for i in subject_pos]

                #Use these subject to predict object
                # find s ,r ,o
                for ix,pos in enumerate(subject_pos):
                    s = subject_token[ix]
                    sub_pos_in = torch.zeros(input_ids.shape)
                    for i in range(pos[0],pos[0]+1):
                        sub_pos_in[0][i] = 1
                    x = [ i.to(self.device) for i in [input_ids,attention_mask,sub_pos_in]]
                    pred_obj_start, pred_obj_end = self.best_model.object_tag(x)
                    # 1* max_seq_len * rel_num
                    pred_obj_start = np.where(F(pred_obj_start) > threshold, 1, 0)
                    pred_obj_end = np.where(F(pred_obj_end) > threshold, 1, 0)

                    for i,rel in id2rel.items():
                        r = rel
                        obj_pos = self._extract_entity(pred_obj_start[:,i],pred_obj_end[:,i])
                        for pos in obj_pos:
                            o = tostr(input_ids_[pos[0]:pos[0]+1])
                            y_.append(' '.join([s,r,o]))

                            decode = lambda x:tokenizer.decode(x.split(' '))
                            ans_triple.append({'subject':decode(s),'relation':r,'object':decode(o)})
                        
                #Summary all
                a += len(y)
                b += len(y_)
                for i in y_:
                    if i in y:
                        tp += 1
        
        recall = float(tp) / a 
        precision = float(tp) / b
        f1_score = float(2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0.

        logger.info("Extract Entity : Precision : {} \t Recall : {} \t F1-Score : {}"\
            .format(precision,recall,f1_score))
        
        #Store the Extracted Result
        with open(filepath, 'w', encoding ='utf-8') as f:
            json.dump(ans_triple,f)
            

    def _extract_entity(self, start, end):
        assert len(start) ==  len(end)
        a = []
        for ix,v in enumerate(start):
            if v:
                j = ix
                while not end[j]:
                    j += 1
                    if not j<len(end):
                        break
                if j == len(end):
                    continue
                a.append((ix,j))
        return a;
    
            
            
