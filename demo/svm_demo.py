from src.svm_sklearn import SVM

# for i in range(5):
#     print('第' + str(i) + '个模型')
#     svm_model = SVM('data/train' + str(i) + '.txt', 'data/stop/stopwords.txt',
#                     'models/svm/' + str(i))
print(',模型训练开始:')
svm_model0 = SVM('data/train0.txt', 'data/stop/stopwords.txt', 'models/svm/0')

svm_model1 = SVM('data/train1.txt', 'data/stop/stopwords.txt', 'models/svm/1')

svm_model2 = SVM('data/train2.txt', 'data/stop/stopwords.txt', 'models/svm/2')

svm_model3 = SVM('data/train3.txt', 'data/stop/stopwords.txt', 'models/svm/3')

svm_model4 = SVM('data/train4.txt', 'data/stop/stopwords.txt', 'models/svm/4')

test = 'data/test-svm.txt'

lines = open(test, 'r', encoding='utf-8').readlines()

for line in lines:
    print(line)
    print(svm_model0.predict(line))
    print(svm_model1.predict(line))
    print(svm_model2.predict(line))
    print(svm_model3.predict(line))
    print(svm_model4.predict(line))