from extract import KeyWordExtractor
from tovec import load_model
import time

if __name__ == "__main__":
    sentence = """悦谷百味 糙米 1kg（无添加 五谷杂粮 含胚芽 东北 粗粮 大米伴侣 粥米搭档 真空装）
【自营极速达·超值精选购翻天】限时9.9元秒杀，品质生活精致选择，点击进入悦谷百味专场》"""
    s2 = """手机","手机","手机通讯","华为 HUAWEI P10 Plus 6GB+64GB 钻雕金 移动联通电信4G手机 双卡双待","wifi双天线设计！徕卡人像摄影！P10徕卡双摄拍照，低至2988元！"""

    keyword_extractor = KeyWordExtractor(topK=7,)
    print(sentence)
    t00 = time.time()
    ret = keyword_extractor.extract_by_tfidf(sentence)
    t01 = time.time()
    print("tfidf:",ret,t01-t00)
    ret = keyword_extractor.extract_by_textrank(sentence)
    t02 = time.time()
    print("textrank:",ret,t02-t01)
    t1 = time.time()
    ret1 = keyword_extractor.extract_by_lsi(sentence)
    t2 = time.time()
    ret2 = keyword_extractor.extract_by_lsi(s2)
    t3 = time.time()
    print("lsi1:",ret1,t2-t1)
    print("lsi2:",ret2,t3-t2)
    print("*"*30)

    ret = keyword_extractor.extract_by_lda(sentence)
    print("lda",ret,time.time()-t3)

    # model = load_model()
    # print(model["手表"])
    