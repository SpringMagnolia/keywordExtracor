import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs
import os
import math
import gensim
from gensim import corpora, models
import pickle
import time

abs_path = os.path.dirname(__file__)

# 结巴加载用户词典
userDict_path = os.path.join(abs_path,r"./词典/all.txt")
jieba.load_userdict(userDict_path)

# 停用词文本
stopwords_path = os.path.join(abs_path,r"./baidu_stopwords.txt")

def get_stopwords_list():
    """返回stopwords列表"""
    stopwords_list = [i.strip()
                      for i in codecs.open(stopwords_path).readlines()]
    return stopwords_list


# 所有的停用词列表
stopwords_list = get_stopwords_list()

# 现有数据的data_path
# data_path = os.path.join(abs_path,r"./data")
data_path = os.path.join(abs_path,r"./data/a.csv")


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


def get_filepath_list(data_path):
    temp_list = os.listdir(data_path)
    #temp_list中是一个个的文件夹

    filepath_list = [os.path.join(data_path, i) for i in temp_list]
    total_filepath_list = []
    for j in filepath_list:
        inner_filepath = os.listdir(j)
        total_filepath_list.extend([os.path.join(j, k) for k in inner_filepath])
    return total_filepath_list


def _load_data(data_path):
    """把所有的data中的文本进行切割，返回doc的列表，里面放的是每个句子的列表"""
    filepath_list = get_filepath_list(data_path)

    doc_list = []
    for filepath in filepath_list:
        filtered_list = []
        for line in codecs.open(filepath).readlines():
            filtered_list.extend(cut_sentence(line))
        doc_list.append(filtered_list)
    return doc_list

def load_data(data_path):
    cache_path = os.path.join(abs_path,"cache/doc_list.cache")
    #如果缓存文件存在
    if os.path.exists(cache_path):
        with open(cache_path,"rb") as f:
            doc_list = pickle.load(f)
    else: #如果不存在
        doc_list  = []
        for line in codecs.open(data_path).readlines():
            doc_list.append(cut_sentence(line))
        with open(cache_path,"wb") as f:
            pickle.dump(doc_list,f)
    return doc_list



class Tfidf:
    def __init__(self,doc_list):
        """
        idf_dict ：现有文本训练出来的idf的值
        default_idf：现有文本中没有出现的词语的idf的值
        keyword_num:要提取的词语的数量
        """
        self.doc_list = doc_list
        self.idf_dict, self.default_idf = self.get_idf_dict()
    
    def get_idf_dict(self):
        '''获取每个词语的idf的值'''
        # TODO 现在根据的data中的内容，不是特别何时的data，可能会导致计算出的tfidf的值不理想，可以使用不同分类的数据分别放在data下一部分来进行
        idf_dict = {}
        doc_count = len(self.doc_list)
        for doc in self.doc_list:
            for word in set(doc):
                idf_dict[word] = idf_dict.get(word, 0)+1
        # 计算每个词语的idf值
        for k, v in idf_dict.items():
            idf_dict[k] = math.log(doc_count/(1+v))  # +1进行平滑，没出现的词语为1
        default_idf = math.log(len(self.doc_list)/1)  # 对于没有出现的词语，其idf为default_idf
        return idf_dict, default_idf

    def get_tf_dict(self,word_list):
        """获取需要提取关键字的文本的tf的值"""
        tf_dict = {}
        for word in word_list:
            tf_dict[word] = tf_dict.get(word, 0)+1

        word_count = len(word_list)

        for k, v in tf_dict.items():
            # print(type(k),k)
            # print(type(v),v)
            tf_dict[k] = v/word_count
        return tf_dict

    def get_tfidf(self,word_list,topK=7):
        #word_list：需要进行提取的文本中词语的列表
        tf_dict = self.get_tf_dict(word_list)
        tfidf_dict = {}
        for word in word_list:
            tf = tf_dict.get(word, 0)
            idf = self.idf_dict.get(word, self.default_idf)
            tfidf_dict[word] = tf*idf

        # 根据tfidf排序，返回前keyword_num个
        ret = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:topK]
        ret_list = [[k, v] for k, v in ret]
        return ret_list


class TextRank(jieba.analyse.TextRank):
    def __init__(self, window=20, word_min_len=2):
        super(TextRank, self).__init__()
        self.span = window  # 窗口大小
        self.word_min_len = word_min_len  # 单词的最小长度
        # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
        self.pos_filt = frozenset(
            ('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns', 'nt', "nw", "nz", "PER", "LOC", "ORG"))

    def pairfilter(self, wp):
        """过滤条件，返回True或者False"""

        if wp.flag == "eng":
            if len(wp.word) <= 2:
                return False

        if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len \
                and wp.word.lower() not in stopwords_list:
            return True


class TopicModel:
    '''
    使用参考：https://radimrehurek.com/gensim/tut1.html
    '''
    def __init__(self, doc_list, topK=5, model="lsi", num_topics=6):
        """
        doc_list :[doc1,doc2] ;doc1:["word1","word2"....]
        topK:返回topK个词语
        model：lda或者是lsi
        num_topics:主题数量
        """

        #下面有大量的pickle保存，为了加速获取数据
        #self.dictionary = corpora.Dictionary(doc_list)  #含有词语和词语次数的字典
        #corpus = [self.dictionary.doc2bow(doc) for doc in doc_list] #bow模型把词句向量化 id-->次数
        #self.tfidf_model = models.TfidfModel(corpus) #计算tfidf的结果，把值进行标准化
        #
        #
        #

        self.dictionary = corpora.Dictionary(doc_list)  #含有词语和词语次数的字典
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list] #bow模型把词句向量化 id-->次数
        self.tfidf_model = models.TfidfModel(corpus) #计算tfidf的结果，把值进行标准化

        self.corpus_tfidf = self.tfidf_model[corpus] # 返回计算的tfid的结果

        self.topK = topK  
        self.num_topics = num_topics
        
        if model == "lsi":
            self.model = self.train_lsi()  
        
        elif model == "lda":
            self.model = self.train_lda()  


        self.word_list = self.get_word_list(doc_list)
        self.wordtopic_dict = self.get_wordtopic(self.word_list)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_word_list(self, doc_list):
        '''获取所有的词语'''
        word_list = []
        [word_list.extend(single_doc_word_list)
         for single_doc_word_list in doc_list]

        return list(set(word_list))

    def get_wordtopic(self, word_list):
        wordtopic_dict = {}
        for word in word_list:
            single_list = [word]
            #计算一个词语的tfidf的值
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            #infer topic distributions on new, unseen documents,
            wordtopic = self.model[wordcorpus]
            wordtopic_dict[word] = wordtopic
        return wordtopic_dict

    # 计算词的分布和文档的分布的相似度，取相似度最高的topK个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]
        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dict = {}
        for k, v in self.wordtopic_dict.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dict[k] = sim

        # 根据sim排序，返回前keyword_num个
        ret = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)[:self.topK]
        ret_list = [[k, v] for k, v in ret]
        return ret_list


class KeyWordExtractor:
    def __init__(self, topK=7,num_topics=19,window=8,word_min_len=2):
        '''
        topK:返回topK个词语
        num_topics:现有语料的主题数
        window：textrank算法中窗口的大小
        word_min_len:最小词语的长度
        '''
        start = time.time()
        self.topK = topK
        self.num_topics = num_topics
        self.doc_list = load_data(data_path)

        #加载tfidf的model
        self.tfidf_model = self._load_tfidf_model()

        #加载textrank的model
        self.textrank_model = self._load_textrank_mode(window=window,word_min_len=word_min_len)

        #加载lsi的model
        self.lsi_model = self._load_lsi_model()

        #加载lda的model
        self.lda_model = self._load_lda_model()
        print("加载模型耗时:{}s".format(time.time()-start))
    
    def _load_lda_model(self):
        '''
        加载lda模型，优先从缓存加载
        '''
        temp_path = r"./cache/lda_mode_k_{}_num_topics_{}.cache".format(self.topK,self.num_topics)
        lda_topic_mode_path = os.path.join(abs_path,temp_path)
        if os.path.exists(lda_topic_mode_path):
            with open(lda_topic_mode_path,"rb") as f:
                topic_model = pickle.load(f)
        else:
            topic_model =  TopicModel(self.doc_list, self.topK, model="lda", num_topics=self.num_topics)
            with open(lda_topic_mode_path,"wb") as f:
                pickle.dump(topic_model,f)
        return topic_model
        
    
    def _load_lsi_model(self):
        '''
        加载lsi模型，优先从缓存加载
        '''
        temp_path = r"./cache/lsi_mode_k_{}_num_topics_{}.cache".format(self.topK,self.num_topics)
        lsi_topic_mode_path = os.path.join(abs_path,temp_path)
        if os.path.exists(lsi_topic_mode_path):
            # print(lsi_topic_mode_path,")"*30)
            with open(lsi_topic_mode_path,"rb") as f:
                topic_model = pickle.load(f)
        else:
            topic_model = TopicModel(self.doc_list, self.topK, model="lsi", num_topics=self.num_topics)
            with open(lsi_topic_mode_path,"wb") as f:
                pickle.dump(topic_model,f)
        return topic_model
        
    
    def _load_tfidf_model(self):
        '''
        加载tfidf模型，优先从缓存加载
        '''
        tfidf_cahche_path = os.path.join(abs_path,"./cache/tfidf_model.cache")
        # print(tfidf_cahche_path,"+"*50)
        if os.path.exists(tfidf_cahche_path):
            with open(tfidf_cahche_path,"rb") as f:
                tfidf_model = pickle.load(f)
        else:
            tfidf_model = Tfidf(self.doc_list)
            with open(tfidf_cahche_path,"wb") as f:
                pickle.dump(tfidf_model,f)
        return tfidf_model

    
    def _load_textrank_mode(self,window=8, word_min_len=2):
        textrank_model = TextRank(window=window, word_min_len=word_min_len)
        return textrank_model


    def extract_by_tfidf(self, text):
        """
        根据tfidf提取关键词
        text:待提取的文本
        topK:前k个，默认5
        return:前k个关键字
        """
        word_list = cut_sentence(text)
        tags = self.tfidf_model.get_tfidf(word_list,topK=self.topK)
        return tags

    def extract_by_textrank(self, text):
        """
        text:带提取的文本
        """
        allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz")
        tags = self.textrank_model.textrank(text, topK=self.topK, withWeight=True, allowPOS=allowPOS, withFlag=False)
        return tags

    def extract_by_lsi(self, text):
        word_list = cut_sentence(text)
        tags = self.lsi_model.get_simword(word_list)
        return tags

    def extract_by_lda(self, text):
        word_list = cut_sentence(text)
        tags = self.lda_model.get_simword(word_list)
        return tags


if __name__ == "__main__":
    # print(get_filepath_list(data_path))
    sentence = '''荣耀10 GT游戏加速 AIS手持夜景 6GB+64GB 幻影蓝全网通 移动联通电信4G 双卡双待 游戏手机
限时优惠2199！荣耀10GT，游戏加速！荣耀爆品特惠，选品质，购荣耀~
选移动，享大流量，不换号购机！'''
    # sentence = '''"数码","读卡器","数码配件","特兰恩（Tralean） Tralean苹果iphone7/6s/plus五合一多功能读卡器安卓手机 玫瑰粉",'''
    # print(jieba.lcut(sentence))
    # print(cut_sentence(sentence))
    keyword_extractor = KeyWordExtractor(topK=7)
    print(sentence)
    ret = keyword_extractor.extract_by_tfidf(sentence)
    print("tfidf:",ret)
    ret = keyword_extractor.extract_by_textrank(sentence)
    print("textrank:",ret)
    ret = keyword_extractor.extract_by_lsi(sentence)
    print("lsi:",ret)
    ret = keyword_extractor.extract_by_lda(sentence)
    print("lda",ret)
