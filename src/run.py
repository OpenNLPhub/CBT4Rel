import config
from generator import CBTDataSet,get_data
from transformers import BertTokenizer
from wrapper import wrapper
import os
import torch
import numpy as np
import random

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(config.BertPath)
    with open(config.RelPath, 'r', encoding = 'utf-8') as f:
        rel_list = [line.strip() for line in f.readlines()]
    test_dataloader = CBTDataSet(config.TestPath, tokenizer, rel_list, config.batch_size)
    model = wrapper(rel_list)
    model_path = config.ModelPath

    if not os.path.exists(model_path):
        train_dataloader = CBTDataSet(config.TrainPath, tokenizer, rel_list, config.batch_size)
        dev_dataloader = CBTDataSet(config.DevPath, tokenizer, rel_list, config.batch_size)
        model.train(train_dataloader, dev_dataloader)
        model.save_model(model_path)
    else:
        model.load_model(model_path)
    
    model.test(test_dataloader)
    data = get_data(config.TestPath)
    model.extract_triple(data, tokenizer, config.ResultPath)
    


    
    