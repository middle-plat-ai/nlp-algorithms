# coding:utf-8

import fasttext
from text_preprocess import clean_text


class FastText(object):
    """
    利用fasttext来对文本进行分裂
    """

    def __init__(self, train_path, test_path, model_path):
        """
        初始化
        :param train_path: 训练数据路径
        :param test_path: 测试数据路径
        :param model_path: 模型保存路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.model = None
        self.clean()
        pass

    def clean(self):
        with open(self.train_path, 'r', encoding='utf8') as train:
            lines = train.readlines()
            train_lines = []
            for line in lines:
                line_list = line.split('__label__')
                l = clean_text(line_list[0]) + '__label__' + line_list[1]
                train_lines.append(l)

        with open(self.train_path, 'w', encoding='utf8') as train:
            train.writelines(train_lines)

        with open(self.test_path, 'r', encoding='utf8') as test:
            lines1 = test.readlines()
            test_lines = []
            for line in lines1:
                line_list = line.split('__label__')
                l = clean_text(line_list[0]) + '__label__' + line_list[1]
                test_lines.append(l)
        with open(self.test_path, 'w', encoding='utf8') as test:
            test.writelines(test_lines)
        pass

    def train(self):
        """
        训练:参数可以针对性修改,进行调优,目前采用的参数都是默认参数,可能不适合具体领域场景
        :return: 无返回值
        """
        self.model = fasttext.supervised(self.train_path, self.model_path, label_prefix="__label__",
                                         dim=300, min_count=2, ws=3, word_ngrams=3, minn=1, maxn=15,
                                         epoch=20, silent=0, bucket=200000)
        test_result = self.model.test(self.test_path)
        print('准确率: ' + str(test_result.precision))

    def predict(self, text):
        """
        预测一条数据,由于fasttext获取的参数是列表,如果只是简单输入字符串,会将字符串按空格拆分组成列表进行推理
        :param text: 待分类的数据
        :return: 分类后的结果
        """
        return self.model.predict([text])

    def load(self, model_path):
        """
        加载训练好的模型
        :param model_path: 训练好的模型路径
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        self.model = fasttext.load_model(model_path, label_prefix='__label__')


if __name__ == '__main__':
    model = FastText('data/train.txt', 'data/test.txt', 'models/fasttext/model')
    model.train()
