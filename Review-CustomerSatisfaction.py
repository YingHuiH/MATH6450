# -*- coding: utf-8 -*-
"""

@author: Yinghui
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:\\Users\\OneDrive\\project 2')


 
df = pd.read_csv('cathay_all.csv') # 读取数据

#了解数据
# 总体评分的频数分布直方图
print(df['rating'].describe())
plt.hist(df['rating'], histtype='bar', rwidth=0.8)    
plt.legend()    
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.title(u'Distribution of Overall Rating')
plt.show()

# 九个评分的频数分布直方图
fig,(ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(nrows=9 , figsize=(9,9))  
ax0.hist(df['rating'], bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
ax1.hist(df['Legroom'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax2.hist(df['Seat comfort'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax3.hist(df['In-flight Entertainment'],bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax4.hist(df['Customer service'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax5.hist(df['Value for money'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax6.hist(df['Cleanliness'], bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax7.hist(df['Check-in and boarding'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
ax8.hist(df['Food and Beverage'], bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)


fig,axes = plt.subplots(3,3,figsize=(9,9))
axes[0,0].hist(df['rating'], bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',facecolor='r',alpha=0.75)   #
axes[0,1].hist(df['Legroom'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[0,2].hist(df['Seat comfort'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[1,0].hist(df['In-flight Entertainment'],bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[1,1].hist(df['Customer service'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[1,2].hist(df['Value for money'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[2,0].hist(df['Cleanliness'], bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[2,1].hist(df['Check-in and boarding'],bins=[0.5,1.5,2.5,3.5,4.5,5.5], rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)
axes[2,2].hist(df['Food and Beverage'], bins=[0.5,1.5,2.5,3.5,4.5,5.5],rwidth = 0.5,normed=1,histtype='bar',alpha=0.75)

axes[0,0].set_title('Overall Rating')
axes[0,1].set_title('Legroom')
axes[0,2].set_title('Seat comfort ')
axes[1,0].set_title('In-flight Entertainment') 
axes[1,1].set_title('Customer service') 
axes[1,2].set_title('Value for money') 
axes[2,0].set_title('Cleanliness ') 
axes[2,1].set_title('Check-in and boarding ') 
axes[2,2].set_title('Food and Beverage ') 
#axes[0,1].legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = -0.2)# 为防止图例覆盖条形图，将图例放置在条形图的外面
fig.subplots_adjust(wspace=0.3,hspace=0.3)  # 调整子图之间的距离
fig.savefig('9RatingHist.png')

print(df.corr())
print(df.corr().min())
fig, ax = plt.subplots(figsize = (16,9))
sns.heatmap(df.corr(),annot=False, vmax=1,vmin = 0, xticklabels= False, yticklabels= False, square=True, cmap="seismic") #"YlGnBu"  "RdBu"
plt.savefig('9RatingCorr.png')






    

import re
import emoji
import codecs
from wordcloud import WordCloud #导入词云的包


################################  开始处理文本数据  #############################

# 丢掉所有的中文评论和过滤emoji
def is_contain_chinese(check_str):
    #判断字符串中是否包含中文
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False
   
    
# 合并所有review data 构成语料库
tmp_1 = list(df['text'])
tmp_2 = list(df['title'])
 
review_text = []

for i in range(0,len(tmp_1)): #
    rt = re.findall('[a-zA-Z]+', tmp_1[i][1:10])
    if  len(rt) == 0:   # 如果开头是中文，那就直接丢了
        print(tmp_1[i])
    elif is_contain_chinese(tmp_1[i]):
        print(tmp_1[i])
    else:
        
        review_text.append( emoji.demojize(tmp_1[i]) )
 
# 没什么事统计一下词频             
r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~]+'
word_list2 = []
for i in range(0,len(review_text)):
    ff = str(review_text[i])
    ff=re.sub(r,' ',ff)
    words=ff.split(' ')
    for word in words:
         word_list2.append(word)
         
#统计频次
tf = {}
for word in word_list2:
    word = word.lower()        
    word = ''.join(word.split())
    if word in tf:
        tf[word] += 1
    else:
        tf[word] = 1
print(tf) 


################################  将所有评论合并成一个txt文件 #############################



#处理成一个评论语料库
review_txt = ' '
for i in range(0,len(review_text)):
    review_txt += review_text[i]


f = codecs.open("review_text_sep.txt",'w','utf-8')    #若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原 #内容之后写入。可修改该模式（'w+','w','wb'等）
for i in range(0,len(review_text)):
    f.write(review_text[i])   #将字符串写入文件中
    f.write('\n')
f.close()



with open("review_text_sep.txt",encoding='utf-8') as f1:
        docs = f1.readlines()    

f2 = codecs.open("review_text_sep_sentence.txt",'w','utf-8')
for i in range(0,len(docs)):
    sentence_list = docs[i].split('.')
    for j in range(0,len(sentence_list)):
        f2.write(sentence_list[j])   #将字符串写入文件中
        f2.write('\n')
f2.close()
f1.close()





################################  绘制评论文本的词云 #############################


f = open(u'review_text.txt','r',encoding='utf-8').read()
#生成一个词云对象
wordcloud = WordCloud(
        background_color="white", #设置背景为白色，默认为黑色
        width=1500,              #设置图片的宽度
        height=960,              #设置图片的高度
        margin=10               #设置图片的边缘
        ).generate(f)
# 绘制图片
plt.imshow(wordcloud)
# 消除坐标轴
plt.axis("off")
# 展示图片
plt.show()

# 保存图片
wordcloud.to_file('ReviewCloud.png')

 

################################  训练词向量 #############################

#coding:utf-8

import sys
import gensim
import sklearn
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.metrics.pairwise import cosine_similarity #计算cosine相似度
import array # 由于argsort(seq)函数调用的包



#读取分词后的数据并打标记，放到x_train供后续索引 
def get_datasest(text_train):
    with open(text_train, 'r',encoding='utf-8') as cf:
        docs = cf.readlines()
        print(len(docs)) 

    x_train = []
    r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~]+'

    for i, text in enumerate(docs):
        text=re.sub(r,' ',text)
        word_list = list(filter(None,text.split(' ')  ))  #过滤空格导致的空字符串
        #print(word_list)
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip() #去掉最后的换行符号
        #print(word_list)
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train



#模型训练
def train(x_train, savefilename,size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=8, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.save(savefilename)
    return model_dm

#实例测试
def test(test_text,model_to_use):
    model_dm = Doc2Vec.load( model_to_use )
    inferred_vector_dm = model_dm.infer_vector(test_text)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=80130)
    return sims

#将文档的list，用指定的训练好的模型转成向量
def get_textvector(model,text_list):

    r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~]+'
    text_vector = []
    
    for text in text_list:
        text=re.sub(r,' ',text)
        word_list = list(filter(None,text.split(' ')  ))  #过滤空格导致的空字符串
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip() #去掉最后的换行符号
        invec = model.infer_vector(word_list, alpha=0.1, min_alpha=0.0001, steps=5)
        text_vector.append(invec)
    return text_vector


#用于获取一个数字list的顺序，返回索引
def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]


def text_to_matrix(text):
    text_matrix = []    
    r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~]+'
    
    for i in range(0,len(text)):
        item =re.sub(r,' ',text[i])
        word_list = list(filter(None,item.split(' ')  ))  #过滤空格导致的空字符串
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip() #去掉最后的换行符号
        text_matrix.append(word_list)
    return text_matrix

    
def load_text_file(file_loc):
    with open(file_loc, 'r',encoding='utf-8') as cf:
        docs = cf.readlines()
    #print(len(docs)) 
    return docs

def calculate_ratio(model_t_u,sentence,pos_set,neg_set):
    x_list = []
    x = 0
    tmp_1= sentence + pos_set
    tmp_2= sentence + neg_set
    #print(tmp_1)
    for tt in range(0,20): #多次循环计算评分得到平均值
        
        tmp_vector_pos = []
        for i in range(0,len(tmp_1)):    
            inferred_vector = model_t_u.infer_vector(tmp_1[i])
            tmp_vector_pos.append(inferred_vector)
        pos_sim_matrix = cosine_similarity(tmp_vector_pos)
        index = argsort(pos_sim_matrix[0])[len(pos_sim_matrix)-2]  #获取max similarity所在的column index
        x_1 = pos_sim_matrix[0][index]
        #print(x_1)
        
        tmp_vector_neg = []
        for i in range(0,len(tmp_2)):    
            inferred_vector = model_t_u.infer_vector(tmp_2[i])
            tmp_vector_neg.append(inferred_vector)
        neg_sim_matrix = cosine_similarity(tmp_vector_neg)
        index = argsort(neg_sim_matrix[0])[len(neg_sim_matrix)-2]
        x_2 = neg_sim_matrix[0][index]

    
        x = 5 + 5 * x_1 -  5 * x_2
        x_list.append(x)

    return np.mean(x_list)


   

def get_ratio_for_sentence(sentence):
    #print(sentence)
    dirloc = 'C:\\Users\\erint_000\\OneDrive\\HKindex\\project 2\\LabelData\\'
    
    pos_set = load_text_file(str(dirloc + 'CustomerExpectation-overall-pos.txt')) #获取每个维度正负向的语料
    pos_set = text_to_matrix(pos_set)
    neg_set = load_text_file(str(dirloc + 'CustomerExpectation-overall-neg.txt'))
    neg_set = text_to_matrix(neg_set)
    CE_overall = calculate_ratio(model,sentence,pos_set,neg_set)

    pos_set = load_text_file(str(dirloc + 'CustomerComplaint-YN-pos.txt'))
    pos_set = text_to_matrix(pos_set)
    neg_set = load_text_file(str(dirloc + 'CustomerComplaint-YN-neg.txt'))
    neg_set = text_to_matrix(neg_set)
    CC_YN = calculate_ratio(model,sentence,pos_set,neg_set)


    pos_set = load_text_file(str(dirloc + 'CustomerLoyalty-repurchase-pos.txt'))
    pos_set = text_to_matrix(pos_set)
    neg_set = load_text_file(str(dirloc + 'CustomerLoyalty-repurchase-neg.txt'))
    neg_set = text_to_matrix(neg_set)
    CL_rep = calculate_ratio(model,sentence,pos_set,neg_set)

    pos_set = load_text_file(str(dirloc + 'CustomerSatisfaction-ConfirmExp-pos.txt'))
    pos_set = text_to_matrix(pos_set)
    neg_set = load_text_file(str(dirloc + 'CustomerSatisfaction-ConfirmExp-neg.txt'))
    neg_set = text_to_matrix(neg_set)
    CS_exp = calculate_ratio(model,sentence,pos_set,neg_set)

    pos_set = load_text_file(str(dirloc + 'PerceivedQuality-food-pos.txt'))
    pos_set = text_to_matrix(pos_set)
    neg_set = load_text_file(str(dirloc + 'PerceivedQuality-food-neg.txt'))
    neg_set = text_to_matrix(neg_set)
    PQ_food = calculate_ratio(model,sentence,pos_set,neg_set)

    pos_set = load_text_file(str(dirloc + 'PerceivedValue-PriceGivenQuality-pos.txt'))
    pos_set = text_to_matrix(pos_set)
    neg_set = load_text_file(str(dirloc + 'PerceivedValue-PriceGivenQuality-neg.txt'))
    neg_set = text_to_matrix(neg_set)
    PV_PgQ = calculate_ratio(model,sentence,pos_set,neg_set)

    ratio = [CE_overall,PQ_food,PV_PgQ,CS_exp,CC_YN,CL_rep]
    
    return ratio
    
def review_to_ratio(review_origin):

    tmp = review_origin.split('.')
    review = []
    for i in range(0,len(tmp)):
        if len(tmp[i]) > 8:  #去掉一些因为乱码造成的短句子
            review.append(tmp[i])
        
    sentence_sep = text_to_matrix(review)
    ratio_matrix = []
    for ss in sentence_sep:
        ratio_matrix.append( get_ratio_for_sentence(ss) )
    
    tmp_1 = pd.DataFrame(ratio_matrix)
    rating_output = []
    for k in range(0,len(tmp_1))   #从矩阵变成一组打分——选择最极端的得分
		if np.min(tmp_1.loc[:,k]) < 10 - np.max(tmp_1.loc[:,k])
			rating_output.append(np.min(tmp_1.loc[:,k]))
		else:
			rating_output.append(np.max(tmp_1.loc[:,k]))
    
    return rating_output




if __name__ == '__main__':

    
    #训练模型
    x_train=get_datasest("review_text_sep_sentence.txt")
    
    model_to_save = 'model_dm_doc2vec_sentence'  #存储训练好的模型以备后续。将一般文本转为向量，比较相似
    model_dm = train(x_train,model_to_save)
    
        
        
    ## 测试看看 用训练好的模型是否可以比较准确区分出文本之间的相似程度
    
    text_list = ['I was in economy class, foods was alright but small portion (2 meals in almost 14 hours flight) but snacks in between 2 meals',
                 'The price is just right, food serve was great and flight attendants wear really hospitable and genuine smiles',
                 'The food was really good and they also serve desserts at the end of mains',
                 'But the flight was a night flight for almost over 11 hours and to my surprise they didn',
                 'The cabin food was not carefully presented but carelessly thrown together and even dumped to my tray instead of gently laying on it',
                 'The food was rubbish too and it took more than an hour to clear the trays',
                 'Quantity and quality of food was poor',
                 'only the food was not that good',
                 'food is ok']

    text_list = ['foods was alright but small portion',
                 'food serve was great',
                 'The food was really good and they also serve desserts at the end of mains',
                 'But the flight was a night flight for almost over 11 hours and to my surprise they didn',
                 'The cabin food was not carefully presented but carelessly thrown together and even dumped to my tray instead of gently laying on it',
                 'The food was rubbish too ',
                 'Quantity and quality of food was poor',
                 'only the food was not that good',
                 'food is ok']
    
    text_vect = get_textvector(model_dm,text_list)
    tmp = cosine_similarity(text_vect)
    print(cosine_similarity(text_vect))
    
    
    model_to_use = 'model_dm_doc2vec_sentence'
    model = Doc2Vec.load( model_to_use )
     
    df['record_ID'] = df.index
    
    new_rating = []
    
    for k_222 in range(0,85): 
        
        re_input_222 = df['text'][k_222]
        item = review_to_ratio(re_input_222 )
        new_rating.append( [df['record_ID'][k_222]] + item )
        
        print(k_222, item)
        
    new_rating_df_222 = pd.DataFrame(new_rating_222)
    new_rating_df_222.columns = ['record_ID','CE_overall','PQ_food','PV_PgQ','CS_exp','CC_YN','CL_rep']
    
    df_new_222 = pd.merge(df_222,new_rating_df_222)   # 把全部评论数据转成了对应得分
    

