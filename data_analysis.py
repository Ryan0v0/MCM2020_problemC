import pandas as pd
import matplotlib.pylab as plt
import string                            #导入字符串模块


# # 数据预处理


tab1 = "./hair_dryer.tsv"
print(tab1)


tab2 = "."

df_hd = pd.read_csv(tab1, sep='\t', header=0)


df_hd.head()

df_hd.describe(include="all")

df_hd.dtypes

df_hd.info()

# # 文本分析
# ## 整体文本分析

#读取文件，并分词
reviews = ''
for review in df_hd['review_body']:
    reviews = reviews + ' ' + review
hist = {}                                 #创建一个空字典，放词频与单词，无序排列
data = []                                 #创建一个空列表，放词频与单词，有序：从多到少
content = reviews.replace('-',' ')       #连字符—用空格代替
words = content.split()                   #字符串按空格分割--分词

# # 保存全部评论
# fw = open("./reviews.txt", 'w')    #将要输出保存的文件地址
# fw.write(reviews)

#迭代处理：将字典变列表，存入数据
for i in range(len(words)):
    words[i] = words[i].strip(string.punctuation)  #去掉标点符号，去掉首尾
    words[i] = words[i].lower()                    #统一大小写
    if words[i] in hist:                          #统计词频与单词
        hist[words[i]] = hist[words[i]] + 1        #不是第一次
    else:
        hist[words[i]] = 1                         #第一次
#print(hist)                                       #打印字典（词频与单词，无序）

# 删除介词
excludes={"at","when", "was","one","had", "it's","than","would","the","and","of","you","a","with","but","as","be","in","or","are", "i", "it", "to", "hair","this", "is", "my", "dryer", "for", "that", "have"}
for word in excludes:
    if word in hist:
        del(hist[word])

#遍历字典
for key, value in hist.items():                    #遍历字典
    temp = [value,key]                              #变量，变量值
    data.append(temp)                               #添加数据
data.sort(reverse=True)                            #排序
for i in range(0,300):
    print(data[i])                                        #打印列表（词频与单词，有序，从多到少）

# #绘制直方图（词频TOP1-10）
# plt.rcParams['font.sans-serif']=['SimHei']      #直方图正常显示中文字体
# for i in range(0,10):
#     plt.bar((data[i][1],),(data[i][0],))
# plt.xlabel('单词')                                   # 显示x轴名称
# plt.ylabel('词频')                                   # 显示y轴名称
# plt.legend('词频直方图')                             #显示图例
# plt.show()                                            #显示作图结果

# #绘制直方图（词频TOP11-20）
# for i in range(10,20):
#     plt.bar((data[i][1],),(data[i][0],))
# plt.legend('直方图')
# plt.xlabel('单词')
# plt.ylabel('词频')
# plt.show()

