from extract import KeyWordExtractor
from tovec import doc2vec_model,word2vec_model,doc_similarity,doc_infer_vector
import time

if __name__ == "__main__":
    s1 = """悦谷百味 糙米 1kg（无添加 五谷杂粮 含胚芽 东北 粗粮 大米伴侣 粥米搭档 真空装）
【自营极速达·超值精选购翻天】限时9.9元秒杀，品质生活精致选择，点击进入悦谷百味专场》"""
    s2 = """手机","手机","手机通讯","华为 HUAWEI P10 Plus 6GB+64GB 钻雕金 移动联通电信4G手机 双卡双待","wifi双天线设计！徕卡人像摄影！P10徕卡双摄拍照，低至2988元！"""
    s3 = """"手机","手机","手机通讯","Apple iPhone 8 Plus (A1864) 256GB 银色 移动联通电信4G手机","选【移动优惠购】新机配新卡，198优质靓号，流量不限量！"""

    #1. 获取关键词
    # keyword_extractor = KeyWordExtractor(topK=7,)
    # print(s1)
    # t00 = time.time()
    # ret = keyword_extractor.extract_by_tfidf(s1)
    # t01 = time.time()
    # print("tfidf:",ret,t01-t00)
    # ret = keyword_extractor.extract_by_textrank(s1)
    # t02 = time.time()
    # print("textrank:",ret,t02-t01)
    # t1 = time.time()
    # ret1 = keyword_extractor.extract_by_lsi(s1)
    # t2 = time.time()
    # ret2 = keyword_extractor.extract_by_lsi(s2)
    # t3 = time.time()
    # print("lsi1:",ret1,t2-t1)
    # print("lsi2:",ret2,t3-t2)
    # print("*"*30)

    # ret = keyword_extractor.extract_by_lda(s1)
    # print("lda",ret,time.time()-t3)

    #2. 获取词向量
    word = "手表"
    model = word2vec_model()
    print("{} 的vector是：{}\n".format(word,model[word]))  #：[ 0.1789252   0.49610293 -0.35405022 ...] 100列

    #3. 获取文档向量
    model = doc2vec_model() #加载模型
    doc_vec = doc_infer_vector(model,s1) #获取向量
    print(doc_vec)  #[-0.08688905 -0.02424988 -0.146502...] 100列

    #4. 获取文档相似度
    #s1是食品，s2是华为手机
    print("s1和s2的相似度为：",doc_similarity(model,s1,s2)) #s1和s2的相似度为： 0.33631673
    
    #s2是华为手机，s3是苹果手机
    print("s2和s3的相似度为：",doc_similarity(model,s3,s2)) #s2和s3的相似度为： 0.62955505