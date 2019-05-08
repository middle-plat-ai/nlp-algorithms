## classification


* 数据集统一采用网上公开的aclImdb电影评论数据,数据分为训练集和测试集,正面评论和负面评论都各12500条,原来的数据每一条评论都写入一个文件,在处理的时候有点麻烦,我把数据全放到一个文件里,类别和数据之间用##分隔.分类的时候,直接按照比例切分数据集,进行训练和测试即可.


### 1) SVM算法

* 运用线性分类模型进行分类, 在文件svm_sklearn.py里.

```python
from src.svm_sklearn import SVMClassifier
svm_model = SVMClassifier('model/svm/model.pkl')
```

### 2) BiLSTM+Attention

* 双向lstm获取句子的表示，然后用attention机制，最后采用简单的全连接层分类，简单的二分类

```python
from src.bilstm_attention_classifier import BiLSTMAttentionClassifier
model = BiLSTMAttentionClassifier('data/quora/GoogleNews-vectors-negative300.bin.gz', 'model/att',
                                      'model/att/config.pkl', train=True)
print(model.predict_result('this is very good movie, i want to watch it again!'))
print(model.predict_result('this is very bad movie, i hate it'))
```