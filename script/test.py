import json
import os
cwd = os.getcwd()

if __name__ == '__main__':
    fp = os.path.join(cwd,'data','WebNLG','dev_.json')
    with open(fp, 'r', encoding = 'utf-8') as f:
        l = json.load(f)
    print(len(l)/32)