# coding:utf-8

from sanic import Sanic
from sanic.response import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import model_selection
from sklearn import svm
import pickle
import pandas as pd
import re


def clean_text(text):
    """
    清理数据,正则方式,去除标点符号等
    :param text:
    :return:
    """
    text = re.sub(r'["\' ?!【】\[\]./%：:&()=，,<>+_；;\-*]+', " ", text)
    return text


class SVM(object):
    """
    这个类,是用svm对文本进行分类.
    1. 用TFIDF计算权重值
    2. 用卡方检验获取特征
    3. 用SVM进行分类训练
    """

    def __init__(self, train_path, stop_path, tf_model_path, chi_model_path, clf_model_path):
        """
        初始化参数
        :param train_path: 训练路径
        :param stop_path: 停用词路径
        :param tf_model_path: tf_idf模型保存路径
        :param chi_model_path: 卡方检验模型保存路径
        :param clf_model_path: SVM模型保存路径
        """
        self.stop_path = stop_path
        self.stop_words = self.init_stopwords()
        self.train_path = train_path
        self.tf_model_path = tf_model_path
        self.chi_model_path = chi_model_path
        self.clf_model_path = clf_model_path

        # 先读取训练好的models,如果读取不到,则重新训练
        self.tf_idf_model, self.chi_model, self.clf_model = self.read_model()
        if self.tf_idf_model is None and self.chi_model is None and self.clf_model is None:
            self.tf_idf_model, self.chi_model, self.clf_model = self.train_model(0.2)

    def predict(self, text):
        """
        根据模型预测某文件的分类
        :param text: 要分类的文本
        :return: 返回分类
        """
        tf_vector = self.tf_idf_model.transform([text])
        chi_vector = self.chi_model.transform(tf_vector)
        out = self.clf_model.predict(chi_vector)
        print('----------推理结果------：', out)
        return out

    def read_model(self):
        """
        读取训练好的models
        :return: 返回读取到的models
        """
        try:
            file = open(self.tf_model_path, "rb")
            tf_idf_model = pickle.load(file)
            file.close()
            file = open(self.chi_model_path, 'rb')
            chi_model = pickle.load(file)
            file.close()
            file = open(self.clf_model_path, 'rb')
            clf_model = pickle.load(file)
            file.close()
        except FileNotFoundError:
            tf_idf_model = None
            chi_model = None
            clf_model = None

        return tf_idf_model, chi_model, clf_model

    def init_stopwords(self):
        """
        初始化停用词典
        :return:
        """
        with open(self.stop_path, 'r', encoding='utf-8') as f:
            stop_words = f.readlines()
        stop_words = [word.strip() for word in stop_words]
        return stop_words

    def train_model(self,test_size):
        """
        训练模型,简单地将生成的TF-IDF数据,chi提取后的特征,以及svm算法模型写入到了磁盘中
        :return: 返回训练好的模型
        """
        data_set = pd.read_table(self.train_path, sep='##', encoding='utf-8', header=None)
        tf_idf_model = TfidfVectorizer(smooth_idf=True, ngram_range=(1, 1), binary=True, use_idf=True, norm='l2',
                                       sublinear_tf=True)
        tf_vectors = tf_idf_model.fit_transform(data_set[1])
        file = open(self.tf_model_path, "wb")
        pickle.dump(tf_idf_model, file)
        file.close()

        chi_model = SelectKBest(chi2, k=5000).fit(tf_vectors, data_set[0])
        chi_features = chi_model.transform(tf_vectors)

        file = open(self.chi_model_path, "wb")
        pickle.dump(chi_model, file)
        file.close()

        x_train, x_test, y_train, y_test = model_selection.train_test_split(chi_features, data_set[0], test_size=test_size,
                                                                            random_state=42, shuffle=True)
        clf_model = svm.SVC(kernel='linear')        # 这里采用的是线性分类模型,如果采用rbf径向基模型,速度会非常慢.
        clf_model.fit(x_train, y_train)
        score = clf_model.score(x_test, y_test)

        print('----测试准确率----:', score)

        file = open(self.clf_model_path, "wb")
        pickle.dump(clf_model, file)
        file.close()
        return tf_idf_model, chi_model, clf_model


if __name__ == '__main__':

    svm_model = SVM('data/aclImdb.txt', 'data/stop/stopwords.txt', 'models/svm/tf_model.pickle',
                    'models/svm/chi_model.pickle',
                    'models/svm/clf_model.pickle')

    # svm_model.train_model()
    # 启动web
    # app = Sanic()
    #
    # @app.route("/predict", methods=['POST', 'GET'])
    # async def predict(request):
    #     """
    #     采用restful接口的形式,获取分类结果
    #     :param request: {
    #                         "text": "待推测文本"
    #                     }
    #     :return:
    #     """
    #     text = request.json.get('text')
    #     text = clean_text(text)
    #     answer = svm_model.predict(text.lower())
    #
    #     ans = answer[0]
    #
    #     if ans == 0:
    #         return json({'category': 'loan'})
    #     elif ans == 1:
    #         return json({'category': 'not_loan'})
    #     else:
    #         return json({'category': 'unknown'})
    #
    #
    # app.run(host="127.0.0.1", port=8000)
