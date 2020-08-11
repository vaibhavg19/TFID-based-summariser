import os
import re
import sys
import math
import nltk
import time
import tfidf
import config
import random
import standard
import operator
import urllib.request
from nltk.stem import porter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words('english'))
wordLemmatizer = WordNetLemmatizer()
stemmer = porter.PorterStemmer()

def lemmetize_words(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(wordLemmatizer.lemmatize(word))
    return lemmatized_words

def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word, tag in pos_tag:
        if tag in config.POS_TAG_LIST:
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb


def sentence_importance(sentence, dict_freq, sentences):
    sentence_score = 0
    sentence = standard.remove_special_chars(str(sentence))
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = pos_tagging(sentence)
    for word in pos_tagged_sentence:
        if word.lower() not in STOPWORDS and word not in STOPWORDS and len(word) > 1:
            word = word.lower()
            word = wordLemmatizer.lemmatize(word)
            sentence_score = sentence_score + tfidf.word_tf_idf(dict_freq, word, sentences, sentence, wordLemmatizer, STOPWORDS)
    return sentence_score

def get_text(text):
    tokenized_sentence = sent_tokenize(text)
    text = standard.remove_special_chars(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in STOPWORDS]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    return tokenized_words, tokenized_sentence

def get_summary(file):
    start = time.time()
    file = open(f"files/{file}" , 'r')
    text = file.read()
    article_text = text
    print(f"[INFO] text: {file.name}")

    print(f"[INFO] formatting text")
    tokenized_words, tokenized_sentence = get_text(article_text)
    tokenized_words = lemmetize_words(tokenized_words)
    word_freq = freq(tokenized_words)
    no_of_sentences = int((config.INFORMATION_WEIGHT * len(tokenized_sentence))/100)
    print(f"[INFO] extracting summary")
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent, word_freq, tokenized_sentence)

        sentence_with_importance[c] = sentenceimp
        c = c + 1

    sentence_with_importance = sorted(
            sentence_with_importance.items(), 
            key=operator.itemgetter(1), 
            reverse=True
        )
    count = 0
    summary = []
    sentence_no = []

    for word_prob in sentence_with_importance:
        if count < no_of_sentences:
            sentence_no.append(word_prob[0])
            count = count + 1
        else:
            break

    sentence_no.sort()
    count = 1
    for sentence in tokenized_sentence:
        if count in sentence_no:
            summary.append(sentence)
        count = count + 1

    summary = " ".join(summary)
    print(f"[INFO] finished in {str(time.time() - start)[:5]} seconds")
    return no_of_sentences, summary

if __name__ == '__main__':
    file_list = os.listdir('files')
    file = random.choice(file_list)
    sent, summary = get_summary(file)
    print(f"[INFO] percentage of information retained: {config.INFORMATION_WEIGHT}%")
    print(f"[INFO] total number of sentences: {sent}")
    print("-"*20)
    print("Summary: ")
    print(summary)