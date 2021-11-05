# 划分出一个小chunk的训练集合，测试脚本任务

data = []
with open('../dataset/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))
    data = lines[:200]
with open('../dataset/train_test.txt', 'w', encoding='utf-8') as f:
    for d in data:
        f.write(d)

with open('../dataset/dev.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = lines[:200]
with open('../dataset/dev_test.txt', 'w', encoding='utf-8') as f:
    for d in data:
        f.write(d)