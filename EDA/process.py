import jionlp as jio
import json
from nlpcda import Homophone, EquivalentChar
from jionlp.util.file_io import read_file_by_line
import pandas as pd

# with open('../dataset/train.json', 'r', encoding='utf8') as f:
#     for line in f:
#         data = json.loads(line)
#
from EDA.tools.homo import HomophoneSubstitution

# smw = Homophone(create_num=3, change_rate=0.3)
# equivalent = EquivalentChar(create_num=3, change_rate=0.3)
# homophone_substitution = HomophoneSubstitution()
# augments = []
# with open('../dataset/train.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     print("原始训练数据共" + str(len(lines)) + "个")
#     idx = 0
#     for line in lines:
#         data = line.split('\t')
#         # res = smw.replace(data[0])
#         # res.extend(equivalent.replace(data[0]))
#         res = homophone_substitution(data[0],homo_ratio=1, augmentation_num=10)  # 同音词替换
#         # res = jio.random_add_delete(data[0])  # 随机增删字符
#         # res = jio.swap_char_position(data[0])  # 邻近文字换位
#         for k in res:
#             temp = [k, data[1], data[2]]
#             augments.append(temp)
#         # if idx > 10:
#         #     break
#         # idx += 1
# with open('../dataset/train_homo_3.txt', 'w', encoding='utf-8') as f:
#     for aug in augments:
#         f.write('\t'.join(aug))

# augments = []
# with open('../dataset/dev.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     print("原始训练数据共" + str(len(lines)) + "个")
#     idx = 0
#     for line in lines:
#         data = line.split('\t')
#         res = smw.replace(data[0])
#         res.extend(equivalent.replace(data[0]))
#         # res = jio.homophone_substitution(data[0])  # 同音词替换
#         # res = jio.random_add_delete(data[0])  # 随机增删字符
#         # res = jio.swap_char_position(data[0])  # 邻近文字换位
#         for k in res:
#             temp = [k, data[1], data[2]]
#             augments.append(temp)
#         # if idx > 10:
#         #     break
#         # idx += 1
# with open('../dataset/dev_homo_2.txt', 'w', encoding='utf-8') as f:
#     for aug in augments:
#         f.write('\t'.join(aug))

def handle_CHIP():
    CHIP_path = '../dataset/CHIP/train.csv'
    data = pd.read_csv(CHIP_path,header=0)
    print(data.describe())
    print(data.head(5))
    data = data.dropna()
    data.to_csv(CHIP_path, encoding='utf-8', index=False, header=False, sep='\t')
    data = read_file_by_line(CHIP_path)
    test_len = int(len(data) * 0.08)
    with open('../dataset/CHIP/train', 'w', encoding='utf-8') as f:
        for text in data[test_len:]:
            print(text)
            texts = text.split('\t')
            f.write('\t'.join(texts[:3]))
            f.write('\n')
    with open('../dataset/CHIP/dev', 'w', encoding='utf-8') as f:
        for text in data[1:test_len]:
            texts = text.split('\t')
            f.write('\t'.join(texts[:3]))
            f.write('\n')
def handle_ATEC():
    ATEC_PATH = '../dataset/train.json'
    data = read_file_by_line(ATEC_PATH)
    test_len = int(len(data) * 0.08)
    with open('../dataset/ATEC/train', 'w', encoding='utf-8') as f:
        for texts in data[test_len:]:
            # texts = json.loads(text, encoding='utf-8')
            f.write(texts['sentence1'] + '\t' + texts['sentence2'] + '\t' + texts['label'])
            f.write('\n')
    with open('../dataset/ATEC/dev', 'w', encoding='utf-8') as f:
        for texts in data[1:test_len]:
            # texts = json.loads(text, encoding='utf-8')
            f.write(texts['sentence1'] + '\t' + texts['sentence2'] + '\t' + texts['label'])
            f.write('\n')
def valid(data):
    idx = 0
    for line in data:
        idx += 1
        try:
            texts = line.split('\t')
            if len(texts) < 3:
                print("{}:{}".format(idx, line))
            if not texts[-1].isnumeric():
                print("{}:{}".format(idx, line))
        except Exception as e:
            print(e)
            print("{}:{}".format(idx, line))


if __name__ == '__main__':
    # handle_CHIP()
    # handle_ATEC()
    # data = read_file_by_line("../dataset/train_all.txt")
    # for text in data:
    #     texts = text.split('\t')
    #     if len(texts) < 3:
    #         print(text)
    # train = read_file_by_line('../dataset/LCQMC/train')
    # dev = read_file_by_line('../dataset/LCQMC/dev')
    #
    # print(len(dev) / len(train))
    # data = read_file_by_line('../dataset/dev_all.txt')
    # new_data = []
    # for line in data:
    #     cnt = line.count("*")
    #     if cnt >= 3:
    #         continue
    #     new_data.append(line)
    # with open('../dataset/dev_all.txt', 'w', encoding='utf-8') as f:
    #     f.writelines('\n'.join(new_data))
    data = read_file_by_line('../dataset/train_augment_data_fix.txt')
    valid(data)
    # idx = 0
    # repair = []
    # preline = ''
    # for line in data:
    #     idx += 1
    #     line = str(line)
    #     ch = line[-1]
    #     if ch == '0' or ch == '1':
    #         if preline != '':
    #             repair.append(preline + '\t' + line)
    #         else:
    #             repair.append(line)
    #         preline = ''
    #     else:
    #         preline = line
    # with open('../dataset/train_augment_data_fix.txt', 'w', encoding='utf-8') as f:
    #     for line in repair:
    #         texts = line.split('\t')
    #         if len(texts) < 3:
    #             print(line)
    #         else:
    #             f.write(line)
    #             f.write('\n')


    # data = pd.read_csv('../dataset/train_augment_data.txt', sep='\t', names=['s1','s2','label'])
    # data.to_csv('../dataset/train_augment_data.csv',index=False, header=False)