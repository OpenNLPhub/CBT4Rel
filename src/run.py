import config
from generator import CBTDataSet
from transformers import BertTokenizer
from wrapper import wrapper
import os

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(config.BertPath)
    with open(config.RelPath, 'r', encoding = 'utf-8') as f:
        rel_list = [line.strip() for line in f.readlines()]
    test_dataloader = CBTDataSet(config.TestPath, tokenizer, rel_list, config.batch_size)
    model = wrapper()
    model_path = config.ModelPath

    if not os.path.exists(model_path):
        train_dataloader = CBTDataSet(config.TrainPath, tokenizer, rel_list, config.batch_size)
        dev_dataloader = CBTDataSet(config.DevPath, tokenizer, rel_list, config.batch_size)
        wrapper.train(train_dataloader,dev_dataloader)
        wrapper.save_model(model_path)
    else:
        wrapper.load_model(model_path)
        
    wrapper.test(test_dataloader)



    
    