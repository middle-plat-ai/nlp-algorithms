from annoy import AnnoyIndex
from sanic import Sanic
from sanic.response import json


class Question:

    def __init__(self, question, category, intention):
        self._intention = intention
        self._question = question
        self._category = category

    @property
    def intention(self):
        return self._intention

    @intention.setter
    def intention(self, value):
        self._intention = value

    @property
    def question(self):
        return self._question

    @question.setter
    def question(self, value):
        self._question = value

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, value):
        self._category = value


class AnnoyCompare:

    def __init__(self, model_path, file_path=None, vector_len=291):
        self.model_path = model_path
        self.vector_len = vector_len
        self.file_path = file_path
        self.question_dict, self.vector_dict = self.preprocess()
        self.annoy_model = self.load()
        if not self.annoy_model:
            self.annoy_model = self.train()
        pass

    def preprocess(self):
        question_dict = dict()
        vector_dict = dict()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line_arr = line.split('\t')
            vector = line_arr[4].split(' ')
            vector = [float(vec) for vec in vector]
            question = Question(line_arr[1], line_arr[2], line_arr[3])
            question_dict[int(line_arr[0])] = question
            vector_dict[int(line_arr[0])] = vector
        return question_dict, vector_dict

    def train(self):

        annoy_model = AnnoyIndex(self.vector_len)  # Length of item vector that will be indexed
        for k, v in self.vector_dict.items():
            annoy_model.add_item(k, v)
        annoy_model.build(100)  # 100 trees
        annoy_model.save(self.model_path)
        return annoy_model

    def load(self):
        annoy_model = AnnoyIndex(self.vector_len)
        try:
            annoy_model.load(self.model_path)  # super fast, will just mmap the file
            return annoy_model
        except FileNotFoundError:
            return None

    def predict(self, vector, num=10):
        ans = self.annoy_model.get_nns_by_vector(vector, num)
        out_ans = [self.question_dict[a] for a in ans]
        return out_ans


app = Sanic()

annoy_compare = AnnoyCompare('question.ann', 'question.txt')


@app.route("/predict", methods=['POST', 'GET'])
async def predict(request):
    """
    采用restful接口的形式,获取分类结果
    :param request: {
                        "text": "待推测文本"
                    }
    :return:
    """
    text = request.json.get('vector')
    vecs = text.split(' ')
    vecs = [float(vec) for vec in vecs]

    ans = annoy_compare.predict(vecs, 3)

    if ans:
        return json(ans)
    else:
        return json([])


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)
