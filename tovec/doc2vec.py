import gensim.models 
from gensim.corpora import WikiCorpus
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import codecs
import os
import jieba
import jieba.posseg as pseg
import numpy as np


abs_path = os.path.dirname(__file__)

# 结巴加载用户词典
userDict_path = os.path.join(abs_path,r"../extract/词典/all.txt")
jieba.load_userdict(userDict_path)

# 停用词文本
stopwords_path = os.path.join(abs_path,r"../extract/baidu_stopwords.txt")

def get_stopwords_list():
    """返回stopwords列表"""
    stopwords_list = [i.strip()
                      for i in codecs.open(stopwords_path).readlines()]
    return stopwords_list


# 所有的停用词列表
stopwords_list = get_stopwords_list()

# 现有数据的data_path
# data_path = os.path.join(abs_path,r"./data")
data_path = os.path.join(abs_path,r"../extract/data/a.csv")


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
        elif seg.flag == "eng":
            if len(seg.word) <= 2:
                continue
            else:
                filtered_words_list.append(seg.word)
        elif seg.flag.startswith("n"):
            filtered_words_list.append(seg.word)
        elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
            filtered_words_list.append(seg.word)
    return filtered_words_list

def build_doc2vec_model():
    document = []
    i = 0
    for line in codecs.open(data_path,"r").readlines():
        words = cut_sentence(line)
        # tag = line.split(",")[:3]
        tag = [str(i)]
        document.append(TaggedDocument(words,tag))
        i+=1
    
    doc2vec_model = Doc2Vec(document,dm=1,vector_size=100,window=8,min_count=1,
                            sample=1e-3, negative=5, workers=8,hs=1,epochs=6)
    
    doc2vec_model_path = os.path.join(abs_path,"doc2vec_model/model.doc2vec")
    doc2vec_model.save(doc2vec_model_path)


def load_model():
    doc2vec_model_path = os.path.join(abs_path,"doc2vec_model/model.doc2vec")
    doc2vec_model = Doc2Vec.load(doc2vec_model_path)
    return doc2vec_model

def doc_similarity(model,text1,text2):
    vec1 = model.infer_vector(cut_sentence(text1))
    vec2 = model.infer_vector(cut_sentence(text2))
    
    # _similarity = 0
    # if vec1 !=0 and vec2 !=0:
    _similarity = (vec1.dot(vec2))/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec1.dot(vec2)))
    return _similarity

def doc_infer_vector(model,text):
    infer_words = cut_sentence(text)
    vec = model.infer_vector(infer_words)
    return vec

def _usage1_infer_vector(text):
    model = load_model()
    infer_words = cut_sentence(text)
    vec = model.infer_vector(infer_words)
    # print(vec)
    return vec

def _usage2_find_exist_most_similar_sentence(text):
    model = load_model()
    infer_words = cut_sentence(text)
    infer_vector = model.infer_vector(infer_words)
    sims = model.docvecs.most_similar([infer_vector],topn=len(model.docvecs))
    
    print('Test Document :{} \n'.format(text))

    print('SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)

    doc_list = codecs.open(data_path,"r").readlines()
    
    for k in range(10):
        print("%s %s: %s\n"%(k,sims[k],doc_list[int(sims[k][0])]))

if __name__ == "__main__":
    #1. 构建模型
    # build_doc2vec_model()

    text = '''相机","运动相机","摄影摄像","GoPro hero7运动相机水下潜水 4K户外直播防水摄像机 官方标配+三向自拍杆+双充电池+64G卡 hero7 black黑色(4K.60帧支持直播）","【11月1日0：00开门红秒杀，立即抢购】【HyperSmooth视频稳定升级】【4K60+12MP高清画质/照片定时器】'''
    #2，获取doc的vector
    # ret =  _usage1_infer_vector(text)
    # print(type(ret))

    #3. 获取用来训练模型的现有语料中和text最相似的doc
    # _usage2_find_exist_most_similar_sentence(text)

    text2 = '''"手机","数据线","手机配件","酷波【两个装】安卓Micro转Type-C转接头 手机数据线/充电线转换头 适用小米8SE/6华为P20荣耀9i一加6三星S9+",""'''

    #4. 获取文档的vector
    model = load_model()
    vec = doc_infer_vector(model,text)
    print(vec)

    #5. 计算文档相似性

    sim = similarity(model,text,text2)
    print(sim)


    

