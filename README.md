# CBT4Rel
Reimplement of CasRel  Using Torch

Paper "A Novel Cascade Binary Tagging Frameword for Relational Triple Extraction" ACL 2020

The [Original code](https://github.com/weizhepei/CasRel) was written in keras



## Requirements

- transformers 3.1.0
- torch 1.5.1+cu101
- tqdm



## Dataset

Using WebNLG DataSet and I preprocess this dataset.

链接:https://pan.baidu.com/s/1RLgcRR1pRXCaBxR5NrA5AQ  密码:8nqn

This Data has been preprocessed



## Usage

**Get the pre-trained Bert Model**

 - Download the [bert-base-uncased](https://huggingface.co/bert-base-uncased)

 - mkdir under data directory

   ```shell
   mkdir data/static
   mv {Default-Path}/bert-base-uncased data/static/
   ```

Note : Please following above shell command, do not use your own path



**Train the Model and Test**

```shell
python src/run.py
```



## Result (updating)

The pure model performance 

|                     |  Acc  | Precision | Recall | F1-score |
| :-----------------: | :---: | :-------: | :----: | :------: |
| subject_start_point | 0.994 |   0.943   | 0.964  |  0.953   |
|  subject_end_point  | 0.994 |   0.946   | 0.962  |  0.954   |
| object_start_point  | 0.999 |   0.923   | 0.766  |  0.837   |
|  object_end_point   | 0.999 |   0.927   | 0.764  |  0.838   |

~~Decoding Part is not finished. QAQ~~

You can check the logger.py under the folder of log for more detail training process.



|         | Precision | Recall | F1-score |
| ------- | --------- | ------ | -------- |
| End2End | 0.662     | 0.517  | 0.577    |
|         |           |        |          |
|         |           |        |          |

In end2end test, model dosen't perform good as it descriped in Paper.

I am still find the problem

## Training Tip

 **! ! !**

**Set epoches a max value, and have enough patient**

In my Training Experiment, I set epoches = 100.

- When epoch equals to 23, the subject tagger starts to recall

- When epoch equal to 31, the subject tagger starts to recall.

- Before this, in validation, the precision, recall and f1-score equal to zero.

