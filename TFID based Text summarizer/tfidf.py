import re
import math
import config
import standard

def tf_score(word, sentence):
    freq_sum = 0
    word_freq_in_sent = 0
    len_sent = len(sentence)
    for word_in_sent in sentence.split():
        if word == word_in_sent:
            word_freq_in_sent = word_freq_in_sent + 1

    tf = word_freq_in_sent / len_sent
    return tf

def idf_score(no_of_sent, word, sentences, wordLemmatizer, STOPWORDS):
    no_of_sent_containing_word = 0
    for sentence in sentences:
        sentence = standard.remove_special_chars(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in STOPWORDS and len(word) > 1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordLemmatizer.lemmatize(word) for word in sentence]

        if word in sentence:
            no_of_sent_containing_word = no_of_sent_containing_word + 1

    idf = math.log10(no_of_sent/no_of_sent_containing_word)
    return idf

def tf_idf_score(tf, idf):
    return tf*idf

def word_tf_idf(dict_freq, word, sentences, sent, wordLemmatizer, STOPWORDS):
    word_tf_idf = [] 
    tf = tf_score(word, sent)
    idf = idf_score(len(sentences), word, sentences, wordLemmatizer, STOPWORDS)
    tf_idf = tf_idf_score(tf, idf)
    return tf_idf