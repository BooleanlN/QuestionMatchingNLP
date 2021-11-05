import jionlp as jio
from LAC import LAC
from collections import defaultdict
from jionlp.util.file_io import read_file_by_line
from paddlenlp import Taskflow
from nlpcda import RandomDeleteChar, CharPositionExchange, baidu_translate
from googletrans import Translator
import os
import json

import time

WORD_PATH = '/Users/jiayi/PycharmProjects/paddle/dataset/'


class SwapNER(object):
    def __init__(self):
        self.entity_dict = defaultdict(dict)
        self._construct_ner_dataset()
        json_str = json.dumps(self.entity_dict, ensure_ascii=False)
        with open('data_set.json', 'w') as json_file:
            json_file.write(json_str)

    def _construct_ner_dataset(self):
        word_info = read_file_by_line(
            os.path.join(WORD_PATH, 'train_all.txt'))
        print("共{}条训练数据".format(len(word_info)))
        for line in word_info[:20000]:
            texts = line.split('\t')
            for text in texts[:2]:
                self.update_entity_dict(text)

    def update_entity_dict(self, text):
        lac = LAC(mode='lac')
        lac_result = lac.run(text)
        for word, ner_type in zip(lac_result[0], lac_result[1]):
            if word in self.entity_dict[ner_type]:
                self.entity_dict[ner_type][word] += 1
            self.entity_dict[ner_type][word] = 1


def replace_ner():
    pass


if __name__ == '__main__':
    # swapNER = SwapNER()
    # text_correction = Taskflow("text_correction")
    # print(text_correction('您好'))
    smw = RandomDeleteChar(create_num=3, change_rate=0.01)
    print( smw.replace("花呗怎么升级额度"))

    smw = CharPositionExchange(create_num=3, change_rate=0.3, char_gram=3, seed=1)
    print(smw.replace("花呗怎么升级额度"))

    print(jio.random_add_delete("花呗怎么升级额度"))

    zh = '天气晴朗，天气很不错，空气很好'
    # 申请你的 appid、secretKey
    # 两遍洗数据法（回来的中文一般和原来不一样，要是一样，就不要了，靠运气？）
    en_s = baidu_translate(content=zh, appid='20211022000979813', secretKey='N7RW184i2eZpDOXuFpkR', t_from='zh', t_to='en')
    time.sleep(1)
    zh_s = baidu_translate(content=en_s, appid='20211022000979813', secretKey='N7RW184i2eZpDOXuFpkR', t_from='en',
                           t_to='zh')
    print(zh_s)