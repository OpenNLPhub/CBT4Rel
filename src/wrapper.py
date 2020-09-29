


import torch
import numpy as np
from transformers import BertModel
import config
import torch.optim as optim
from model import CBT
from utils import focal_loss
from log import logger
from copy import deepcopy
from utils import confusion_matrix,evaluate
class wrapper(object):

    def __init__(self,rel_list):
        self.rel_list = rel_list

        self.device = torch.device( 'cuda:1' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel.from_pretrained(config.BertPath)
        
        self.model =  CBT(self.bert,len(self.rel_list)).to(self.device)
        self.epoches = 100
        self.lr = 1e-5

        self.best_model = None
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