
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler




# Statistical features
def statistics(data):
    print("Processing basic featues ...")
    num_title_words, num_title_token, num_title_char, num_title_sent = [], [], [], []
    num_body_words, num_body_token, num_body_para, num_body_sent = [], [], [], []
    for title in data['title']:
        num_title_words.append(len(title.split()))
        tokens = nltk.word_tokenize(title)
        num_title_token.append(len(tokens))
        num_title_char.append(len(title))
        sentences = nltk.tokenize.sent_tokenize(title, language='english')
        num_title_sent.append(len(sentences))
    for body in data['usertext']:
        temp_words, temp_token, temp_sent = 0, 0, 0
        for para in body:
            temp_words += len(para.split())
            temp_token += len(nltk.word_tokenize(para))
            temp_sent += len(nltk.tokenize.sent_tokenize(para, language='english'))
        num_body_words.append(temp_words)
        num_body_token.append(temp_token)
        num_body_sent.append(temp_sent)
        num_body_para.append(len(body))
    features = {'title_words': num_title_words, 'title_token': num_title_token, 'title_char': num_title_char,
                'title_sent': num_title_sent, 'body_words': num_body_words, 'body_token': num_body_token,
                'body_sent': num_body_sent, 'body_para': num_body_para}
    return pd.DataFrame(features, columns=['title_words', 'title_token', 'title_char', 'title_sent',
                                           'body_words', 'body_token', 'body_sent', 'body_para'])

# POS Featurs
def get_all_tags(data):
    print("Processing POS features ...")
    tags_all = []
    for title in data['title']:
        tagged_text = nltk.pos_tag(nltk.word_tokenize(title))
        for word, tag in tagged_text:
            if tag not in tags_all:
                tags_all.append(tag)
    for body in data['usertext']:
        for para in body:
            tagged_text = nltk.pos_tag(nltk.word_tokenize(para))
            for word, tag in tagged_text:
                if tag not in tags_all:
                    tags_all.append(tag)
    return tags_all


def pos(data, tags_all):
    tag_dict, tag_count, tag_count_body = {}, {}, {}
    for tag in tqdm(tags_all):
        tag_dict[tag] = 0
        tag_count[tag] = []
        tag_count_body[tag] = []
    for title in tqdm(data['title']):
        tagged_text = nltk.pos_tag(nltk.word_tokenize(title))
        for word, tag in tagged_text:
            tag_dict[tag] += 1
        for count, tag in zip(tag_dict.values(), tag_dict.keys()):
            tag_count[tag].append(count)
    for tag in tags_all:
        tag_dict[tag] = 0
    for body in tqdm(data['usertext']):
        for para in body:
            tagged_text = nltk.pos_tag(nltk.word_tokenize(para))
            for word, tag in tagged_text:
                tag_dict[tag] += 1
        for count, tag in zip(tag_dict.values(), tag_dict.keys()):
            tag_count_body[tag].append(count)
    return pd.concat((pd.DataFrame(tag_count, index=None), pd.DataFrame(tag_count_body, index=None)), axis=1)

# TF-IDF Features
def tfidf(data):
    print("Processing TF-IDF features ...")
    X = []
    for t, b in zip(data['title'], data['usertext']):
        X.append(t + ' ' + b)
    count_vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=50)
    X_counts = count_vect.fit_transform(X)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    return pd.DataFrame(X_tfidf.todense())

# Topic Features
def topics(data, topic_num):
    print("Processing Topics features ...")
    def cleaning(article):
        punctuation = set(string.punctuation)
        lemmatize = WordNetLemmatizer()
        one = " ".join([i for i in article.lower().split() if i not in stopwords])
        two = "".join(i for i in one if i not in punctuation)
        three = " ".join(lemmatize.lemmatize(i) for i in two.lower().split())
        return three

    def pred_new(doc):
        one = cleaning(doc).split()
        two = dictionary.doc2bow(one)
        return two

    def load_title_body(data):
        text =[]
        for i in range(len(data["y"])):
            temp = str(data["title"][i])[2:-2]
            for j in data["usertext"][i]:
                temp = temp + ' ' + str(j)[2:-2]
            text.append(temp)
        return text

    stopwords = set(nltk.corpus.stopwords.words('english'))
    text_all = load_title_body(data)
    df = pd.DataFrame({'text': text_all}, index=None)
    text = df.applymap(cleaning)['text']
    text_list = []
    for t in text:
        temp = t.split()
        text_list.append([i for i in temp if i not in stopwords])

    dictionary = corpora.Dictionary(text_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
    ldamodel = LdaModel(doc_term_matrix, num_topics=topic_num, id2word = dictionary, passes=50)
    probs = []
    for text in text_all:
        prob = ldamodel[(pred_new(text))]
        d = dict(prob)
        for i in range(topic_num):
            if i not in d.keys():
                d[i] = 0
        temp = []
        for i in range(topic_num):
            temp.append(d[i])
        probs.append(temp)
    return pd.DataFrame(probs, index=None)


# N-grams Features
def ngram(data, window, limit):
    # Create a CountVectorizer object
    vectorizer = CountVectorizer(ngram_range=(window, window), max_features=limit)

    # Fit and transform the data using the vectorizer
    ngram_features = vectorizer.fit_transform(data)

    # Get the feature names
    feature_names = vectorizer.get_feature_names()

    return ngram_features, feature_names

