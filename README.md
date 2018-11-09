## classification


* 数据集统一采用网上公开的aclImdb电影评论数据,数据分为训练集和测试集,正面评论和负面评论都各12500条,原来的数据每一条评论都写入一个文件,在处理的时候有点麻烦,我把数据全放到一个文件里,类别和数据之间用##分隔.分类的时候,直接按照比例切分数据集,进行训练和测试即可.


### 1) SVM算法

* 运用线性分类模型进行分类.

```python
svm_model = SVM('data/aclImdb.txt', 'data/stop/stopwords.txt', 'models/svm/tf_model.pickle',
                    'models/svm/chi_model.pickle',
                    'models/svm/clf_model.pickle')
```

