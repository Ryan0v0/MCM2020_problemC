import pandas as pd
import matplotlib.pylab as plt
import string                            #导入字符串模块
import numpy as np
import math
import csv

from itertools import chain

tab1 = "./hair_dryer.tsv"

df = pd.read_csv(tab1, sep='\t', header=0)

df = df[~(df['vine'].str.contains("N") & df['verified_purchase'].str.contains("N"))]

df = df.groupby('product_parent').filter(lambda x: len(x) > 1)

gp_pp = df.groupby('product_parent')

# # 选特定的一组
# print(gp_pp.get_group(732252283)['star_rating'])

# # 遍历
# for item in gp_pp:
#     print(item[0])
#     print(item[1]['star_rating'])

# gp_pp[['star_rating', 'helpful_votes', 'total_votes']].sum()

gp_pp[['star_rating', 'helpful_votes', 'total_votes']].sum().sort_values(by='helpful_votes',ascending=False).head(10)


# ------------------------------------------------------------------------------------------------------------------------------
# 预处理完成


# 定义功能函数

import re
import nltk

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#import nltk resources
resources = ["wordnet", "stopwords", "punkt", \
             "averaged_perceptron_tagger", "maxent_treebank_pos_tagger"]

for resource in resources:
    try:
        nltk.data.find("tokenizers/" + resource)
    except LookupError:
        nltk.download(resource)

#create Lemmatizer object
lemma = WordNetLemmatizer()

def lemmatize_word(tagged_token):
    """ Returns lemmatized word given its tag"""
    root = []
    for token in tagged_token:
        tag = token[1][0]
        word = token[0]
        if tag.startswith('J'):
            root.append(lemma.lemmatize(word, wordnet.ADJ))
        elif tag.startswith('V'):
            root.append(lemma.lemmatize(word, wordnet.VERB))
        elif tag.startswith('N'):
            root.append(lemma.lemmatize(word, wordnet.NOUN))
        elif tag.startswith('R'):
            root.append(lemma.lemmatize(word, wordnet.ADV))
        else:          
            root.append(word)
    return root

def lemmatize_doc(document):
    """ Tags words then returns sentence with lemmatized words"""
    lemmatized_list = []
    tokenized_sent = sent_tokenize(document)
    for sentence in tokenized_sent:
        no_punctuation = re.sub(r"[`'\",.!?()]", " ", sentence)
        tokenized_word = word_tokenize(no_punctuation)
        tagged_token = pos_tag(tokenized_word)
        lemmatized = lemmatize_word(tagged_token)
        lemmatized_list.extend(lemmatized)
    return " ".join(lemmatized_list)


from unicodedata import normalize

remove_accent = lambda text: normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

wordnet_lemmatizer = WordNetLemmatizer()

adjective_tags = ['JJ','JJR','JJS']
# -------------------------------------------------------------------------------------------------------------------------------


# product_parents = [758099411, 47684938, 732252283, 127343313, 694290590, 357308868, 194533684, 531479992, 614083399]

product_parents = [758099411, 47684938, 732252283, 127343313]
num_PSW = []
score_S = [-0.5, -1, 0, 0.5, 1]
num_PSR = [] * len(product_parents) 
score_PSW = []

keywords = [] # 需要考虑的词集
scores = [] # 关键词得分
keywords_num = 20

del_word = ['ta', 'diff', 'low', 'ur', 'oo']
change_word = {'yr': 'year', 'hr': 'hour', 'mo': 'moment', 'ra': 'rate adaptation'}

csvFile = open('keyword_20_noun.csv', 'r')
reader = csv.reader(csvFile)
for item in reader:
    # 忽略第一行
    if reader.line_num == 1:
        continue
    if reader.line_num > keywords_num + 1:
        break
    keywords.append(item[0])
    scores.append(float(item[1]))



id_i = 0
for id in product_parents:
    num_PSR.append(list())
    num_PSW.append(list())
    df_pp = gp_pp.get_group(id)

    for star in range(1, 6):
        star_scores = [0] * keywords_num
        df = df_pp[df_pp['star_rating'] == star]
        num_PSR[id_i].append(len(df))
        

        pattern = r"\&\#[0-9]+\;"
        df["preprocessed"] = df["review_body"].str.replace(pat=pattern, repl="", regex=True)
        df["preprocessed"] = df["preprocessed"].apply(lambda row: lemmatize_doc(row))
        df["preprocessed"] = df["preprocessed"].apply(remove_accent)
        pattern = r"[^\w\s]"
        df["preprocessed"] = df["preprocessed"].str.replace(pat=pattern, repl=" ", regex=True)
        df["preprocessed"] = df["preprocessed"].str.lower()
        pattern = r"[\s]+"
        df["preprocessed"] = df["preprocessed"].str.replace(pat=pattern, repl=" ", regex=True)
        corpora = df["preprocessed"].values
        tokenized = [corpus.split(" ") for corpus in corpora]
        for review in tokenized:
            while '' in review:
                review.remove('')
        for text in tokenized:
            selected = [0] * keywords_num
            POS_tag = nltk.pos_tag(text)
            lemmatized_text = []
            for word in POS_tag:
                if word[1] in adjective_tags:
                    lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
                else:
                    lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
            POS_tag = nltk.pos_tag(lemmatized_text)
            stopwords = []
            wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 

            for word in POS_tag:
                if word[1] not in wanted_POS:
                    stopwords.append(word[0])

            punctuations = list(str(string.punctuation))

            stopwords = stopwords + punctuations
            stopword_file = open("long_stopwords.txt", "r")

            lots_of_stopwords = []

            for line in stopword_file.readlines():
                lots_of_stopwords.append(str(line.strip()))

            stopwords_plus = []
            stopwords_plus = stopwords + lots_of_stopwords
            stopwords_plus = set(stopwords_plus)

            processed_text = []
            for word in lemmatized_text:
                if word not in stopwords_plus:
                    processed_text.append(word)

            vocabulary = list(set(processed_text))
            phrases = []
            phrase = " "
            for word in lemmatized_text:
                
                if word in stopwords_plus:
                    if phrase!= " ":
                        phrases.append(str(phrase).strip().split())
                    phrase = " "
                elif word not in stopwords_plus:
                    phrase+=str(word)
                    phrase+=" "

            for word in vocabulary:
                for phrase in phrases:
                    if (word in phrase) and (word in phrases) and (len(phrase)>1):
                        #if len(phrase)>1 then the current phrase is multi-worded.
                        #if the word in vocabulary is present in unique_phrases as a single-word-phrase
                        # and at the same time present as a word within a multi-worded phrase,
                        # then I will remove the single-word-phrase from the list.
                        phrases.remove([word])
            
            for phrase in phrases:
                word = ' '
                for word in phrase:
                    if (word in keywords) and (selected[keywords.index(word)] != 1):
                        star_scores[keywords.index(word)] = star_scores[keywords.index(word)] + 1 - selected[keywords.index(word)]
                        selected[keywords.index(word)] = 1
                    else:
                        for i in range(0, len(keywords)):
                            if word in change_word.keys():
                                word = change_word[word]
                            if (word in keywords[i]) and (word not in del_word) and (selected[i] == 0):
                                star_scores[i] = star_scores[i] + 0.5
                                selected[i] = 0.5

        num_PSW[id_i].append(star_scores)
    id_i = id_i + 1

def sigmoid(x, h = 1, v = 1):
    return v * 1 / (1 + math.exp(- x / h * 6))

score_PSW_sigmoid = []
for i in range(len(product_parents)):
    score_PSW.append(list())
    score_PSW_sigmoid.append(list())
    for j in range(keywords_num):
        res = 0
        for k in range(5):
            res = res + score_S[k] * num_PSW[i][k][j] / math.sqrt(num_PSR[i][k])
        score_PSW[i].append(res)
        score_PSW_sigmoid[i].append(sigmoid(res, 1.23, 9))

B = []
for i in range(keywords_num):
    B.append(list())
    for j in range(len(product_parents)):
        B[i].append(list())
        for k in range(len(product_parents)):
            B[i][j].append(score_PSW_sigmoid[j][i] /  score_PSW_sigmoid[k][i])
            
def stretch(x):
    return (x - 1) * 4 + 1

A = []
for i in range(0, keywords_num):
    A.append(list())
    for j in range(0, keywords_num):
        res = scores[i] / scores[j]
        if res > 1:
            A[i].append(stretch(res))
        elif res < 1:
            A[i].append(1/stretch(scores[j] / scores[i]))
        else:
            A[i].append(res)

class AHP:
    def __init__(self, criteria, b):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria
        self.b = b
        self.num_criteria = criteria.shape[0]
        self.num_project = b[0].shape[0]

    def cal_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, '不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError('不是反互对称矩阵')

        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('无法判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n]
        return max_eigen, CR, eigen

    def run(self):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        print('准则层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_eigen))

        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(eigen)

        pd_print = pd.DataFrame(eigen_list,
                                index=['准则' + str(i) for i in range(self.num_criteria)],
                                columns=['方案' + str(i) for i in range(self.num_project)],
                                )
        pd_print.loc[:, '最大特征值'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1
        print('方案层')
        print(pd_print)

        # 目标层
        obj = np.dot(criteria_eigen.reshape(1, -1), np.array(eigen_list))
        print('\n目标层', obj)
        print('最优选择是方案{}'.format(np.argmax(obj)))
        return obj


# 准则重要性矩阵
criteria = np.array(A)

plan = []

for b in B:
    plan.append(np.array(b))

a = AHP(criteria, plan).run()








