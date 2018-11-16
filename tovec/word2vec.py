import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs
import os
import math
import gensim
from gensim import corpora, models
from gensim.models import word2vec

abs_path = os.path.dirname(__file__)

# 结巴加载用户词典
userDict_path = os.path.join(abs_path,r"../extract/词典/all.txt")
jieba.load_userdict(userDict_path)


# 停用词文本
stopwords_path = os.path.join(abs_path,r"../extract/baidu_stopwords.txt")

def get_stopwords_list():
    """返回stopwords列表"""
    stopwords_list = [i.strip() for i in codecs.open(stopwords_path).readlines()]
    return stopwords_list
# 所有的停用词列表
stopwords_list = get_stopwords_list()

# 现有数据的data_path
data_path = os.path.join(abs_path,r"./a.csv")

def cut_sentence(sentence):
    """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
    # print(sentence,"*"*100)
    # eg:[pair('今天', 't'), pair('有', 'd'), pair('雾', 'n'), pair('霾', 'g')]
    seg_list = pseg.lcut(sentence)
    seg_list = [i for i in seg_list if i.flag not in stopwords_list]
    filtered_words_list = []
    for seg in seg_list:
        # print(seg)
        if len(seg.word) <= 1:
            continue
        if seg.flag == "eng":
            if len(seg.word) <= 2:
                continue
        elif seg.flag.startswith("n"):
            filtered_words_list.append(seg.word)
        elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
            filtered_words_list.append(seg.word)
    return filtered_words_list

def build_text8(data_path):
    '''
    构造分好词的字符串，每一行是一个产品的词语，保存到txt中
    '''
    lines = codecs.open(data_path).readlines()
    with open("text8.txt","a") as f:
        for line in lines:
            temp_str = " ".join(cut_sentence(line)) #用空格隔开
            f.write(temp_str)
            f.write("\n")
    

def build_word2vec_model():
    input_file = os.path.join(abs_path, r"text8.txt")
    sentences = word2vec.Text8Corpus(input_file)
    model = word2vec.Word2Vec(sentences, sg=1, size=100, window=5, min_count=1, 
                              negative=1, sample=0.001, hs=1,workers=40)

    model.save("./sku_word.model")
    model.wv.save_word2vec_format("./sku.wor2vec.txt")

    #load 的时候只需要
    #model = word2vec.Word2Vec.load("./sku_word.model")
    #model=gensim.models.KeyedVectors.load_word2vec_format("./sku.wor2vec.txt")
    return model

def load_model(): 
    model_path = "./sku_word.model"
    model_path = os.path.join(abs_path,model_path)
    model = word2vec.Word2Vec.load(model_path)
    #model=gensim.models.KeyedVectors.load_word2vec_format("./sku.wor2vec.txt")
    return model


if __name__ == "__main__":
    # build_text8(data_path)
    # model = build_word2vec_model()
    model = load_model()
    print("'华为'和'荣耀'的相似度为：",model.similarity('荣耀', '华为'))
    print("和'蓝牙耳机'最相似的词语是：",model.most_similar(['蓝牙耳机']))
    print()
    print("和'华为'最相似的词语是：",model.most_similar(['华为']))
    print()
    print("和'牛仔裤'最相似的词语是：",model.most_similar(['牛仔裤']))

