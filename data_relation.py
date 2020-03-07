import csv
import re
import pandas as pd
import matplotlib.pylab as plt
import string                            #导入字符串模块
import nltk

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

from unicodedata import normalize
import numpy as np



csvFile = open('word_n_standard.csv', 'r')
reader = csv.reader(csvFile)
standard_word = [] # 需要考虑的词集
noun_n = 200
source = [] # 存放产品分级的词频数组
source_p = [] # 存放产品的词频数组



for item in reader:
    # 忽略第一行
    if reader.line_num == 1:
        continue
    if reader.line_num > noun_n:
        break
    standard_word.append(item[0])
    
# print(standard_word)

## 进行分产品词频处理

tab1 = "./hair_dryer.tsv"
tab2 = "."
df_hd = pd.read_csv(tab1, sep='\t', header=0)
df_hd.head()
df_hd = df_hd[~(df_hd['vine'].str.contains("N") & df_hd['verified_purchase'].str.contains("N"))]
df_hd = df_hd[(df_hd['total_votes'] > 0) & (df_hd['helpful_votes'] > 0)]

df_hd = df_hd.groupby('product_parent').filter(lambda x: len(x) > 1)

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

# 定义处理函数
def process_data(df_pp):
    pattern = r"\&\#[0-9]+\;"
    df_pp["preprocessed"] = df_pp["review_body"].str.replace(pat=pattern, repl="", regex=True)
    
    df_pp["preprocessed"] = df_pp["preprocessed"].apply(lambda row: lemmatize_doc(row))
    
    remove_accent = lambda text: normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    df_pp["preprocessed"] = df_pp["preprocessed"].apply(remove_accent)
    
    pattern = r"[^\w\s]"
    df_pp["preprocessed"] = df_pp["preprocessed"].str.replace(pat=pattern, repl=" ", regex=True)

    df_pp["preprocessed"] = df_pp["preprocessed"].str.lower()

    pattern = r"[\s]+"

    df_pp["preprocessed"] = df_pp["preprocessed"].str.replace(pat=pattern, repl=" ", regex=True)
    
    corpora = df_pp["preprocessed"].values
    tokenized = [corpus.split(" ") for corpus in corpora]

    return tokenized


# 遍历分组
for key, group in df_hd.groupby('product_parent'):
    product_r = list()
    # 遍历星级
    for star in range(1, 6):
        star_r = [0] * noun_n
        # df_pp 代表当前产品某一星级的数据
        df_pp = group[group['star_rating'] == star]
        tokenized = process_data(df_pp)
        for review in tokenized:
            for word in review:
                if word in standard_word:
                    star_r[standard_word.index(word)] = star_r[standard_word.index(word)] + 1
        product_r.append(star_r)

    source.append(product_r)

for product_r in source:
    source_p.append(np.sum(product_r[0], product_r[1], product_r[2], product_r[3], product_r[4]))
    

