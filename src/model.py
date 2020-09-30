
import torch
import torch.nn as nn

class CBT(nn.Module):

    def __init__ (self, bert,rel_num):
        super(CBT,self).__init__()
        self.bert = bert
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim

        self.subject_start_tagger = nn.Linear(self.hidden_dim, 1)
        self.subject_end_tagger = nn.Linear(self.hidden_dim, 1)

        self.object_start_tagger = nn.Linear(self.hidden_dim, rel_num)
        self.object_end_tagger = nn.Linear(self.hidden_dim, rel_num)

    def forward(self,x):
        input_ids, attention_mask, sub_pos = x;
        # input_ids, attention_mask, sub_pos : batch_size * max_seq_len
        xx = self.bert(input_ids = input_ids, attention_mask = attention_mask)[0]
        # batch_size * max_seq_len * word_emb_dim

        sub_start = torch.sigmoid(self.subject_start_tagger(xx))
        sub_end = torch.sigmoid(self.subject_end_tagger(xx))

        sub_attention_mask = sub_pos.unsqueeze(-1).expand(-1,-1,self.hidden_dim)
        # batch_size * max_seq_len * word_emb_dim

        sub_feature = torch.sum(sub_attention_mask * xx,axis = 1) \
            / torch.sum(sub_attention_mask ,axis = 1)
        # batch_size * word_emb_dim / batch_size * word_emb_dim
        
        sub_feature = sub_feature.unsqueeze(1).expand(-1,xx.shape[1],-1)

        token_feature = xx + sub_feature

        obj_end = torch.sigmoid(self.object_end_tagger(token_feature))
        obj_start = torch.sigmoid(self.object_start_tagger(token_feature))
        # batch_size * max_seq_len * rel_num

        F = lambda x : x.squeeze(-1)
        # batch_size * max_seq_len 
        return F(sub_start), F(sub_end), obj_start, obj_end

    
    # add these independent method
    # When extract triples, extract subject independently at first 
    # and then extract object and relationships

    def subject_tag(self,x):
        input_ids, attention_mask = x
        xx = self.bert(input_ids = input_ids, attention_mask = attention_mask)[0]
        # batch_size * max_seq_len * word_emb_dim

        sub_start = torch.sigmoid(self.subject_start_tagger(xx)).squeeze(-1)
        sub_end = torch.sigmoid(self.subject_end_tagger(xx)).squeeze(-1)
        
        return sub_start,sub_end

    def object_tag(self,x):
        
        input_ids, attention_mask, sub_pos = x
        xx = self.bert(input_ids = input_ids, attention_mask = attention_mask)[0]

        sub_attention_mask = sub_pos.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

        sub_feature = torch.sum(sub_attention_mask * xx, axis = 1)\
            / torch.sum(sub_attention_mask, axis = 1)

        sub_feature = sub_feature.unsqueeze(1).expand(-1, xx.shape[1], -1)
        token_feature = xx + sub_feature

        obj_end = torch.sigmoid(self.object_end_tagger(token_feature))
        obj_start = torch.sigmoid(self.object_start_tagger(token_feature))

        return obj_start,obj_end




