'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-27 15:13:44
 * @desc 
'''
import os
cwd = os.getcwd()

''' -------------- Path Config -------------- '''

BertPath = os.path.join(cwd,'data','static','bert-base-uncased')

RelPath = os.path.join(cwd, 'data', 'WebNLG', 'relation_type.txt')
TrainPath = os.path.join(cwd, 'data', 'WebNLG', 'train.json')
DevPath = os.path.join(cwd, 'data', 'WebNLG', 'dev_.json')
TestPath = os.path.join(cwd, 'data', 'WebNLG', 'test.json')

ModelPath = os.path.join(cwd, 'data', 'model', 'cbt.pth')

batch_size = 32