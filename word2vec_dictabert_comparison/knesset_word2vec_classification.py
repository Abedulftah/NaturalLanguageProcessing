import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import random
random.seed(42)


def tokenization(sentence):
    hebrew_alphabet = {'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק',
                       'ר', 'ש', 'ת'}
    words = sentence.split()
    dont_insert = True
    filtered_words = []
    for word in words:
        for char in word:
            if char in hebrew_alphabet:
                dont_insert = False
                break
        if not dont_insert:
            filtered_words.append(word)

    return filtered_words


def get_excel(input_path):
    # ptv committee, ptm plenary.
    try:
        excel = pd.read_csv(os.path.join(input_path), encoding='utf-8')
    except Exception as e:
        print(f"Couldn't read the csv file, {e}")
        return None, None, None
    corpus_tokenized = []
    committee = []
    plenary = []
    try:
        for index, row in excel.iterrows():
            sentence_text_value = row['sentence_text']
            type_column_value = row['protocol_type']
            if type_column_value == 'committee':
                committee.append(sentence_text_value)
            else:
                plenary.append(sentence_text_value)
        for sentence in committee:
            toAppend = tokenization(sentence)
            if len(toAppend) != 0:
                corpus_tokenized.append(toAppend)
        for sentence in plenary:
            toAppend = tokenization(sentence)
            if len(toAppend) != 0:
                corpus_tokenized.append(toAppend)
    except Exception as e:
        print(f"Couldn't iterate over the csv file, {e}")
        return None, None, None
    return corpus_tokenized, committee, plenary


def down_sampling(c_chunks, p_chunks):
    if len(c_chunks) < len(p_chunks):
        return c_chunks, random.sample(p_chunks, len(c_chunks))
    return random.sample(c_chunks, len(p_chunks)), p_chunks


def make_chunks(committee, plenary, chunk_size):
    committee_chunks, plenary_chunks, chunk = [], [], []
    # we took every 5 sentences to make it as a chunk for committee.
    for i in range(len(committee)):
        chunk.append(committee[i])
        if (i + 1) % chunk_size == 0:
            committee_chunks.append(" ".join(chunk.copy()))
            chunk.clear()

    chunk.clear()
    # we took every 5 sentences to make it as a chunk for plenary.
    for i in range(len(plenary)):
        chunk.append(plenary[i])
        if (i + 1) % chunk_size == 0:
            plenary_chunks.append(" ".join(chunk.copy()))
            chunk.clear()
    return down_sampling(committee_chunks, plenary_chunks)


def sentence_embeddings(word_vectors, corpus):
    embeddings = []
    for sentence in corpus:
        summing, k = 0, 0
        sentence_list = sentence.split()
        for word in sentence_list:
            if word in word_vectors:
                summing += word_vectors[word]
                k += 1
        if k != 0:
            embeddings.append([sentence, summing / k])
    return embeddings


def classification_knn(word_vectors, plenary, committee, chunk_size):
    committee_chunks, plenary_chunks = make_chunks(committee, plenary, chunk_size)
    committee_chunks = sentence_embeddings(word_vectors, committee_chunks)
    plenary_chunks = sentence_embeddings(word_vectors, plenary_chunks)
    committee_labels = np.zeros(len(committee_chunks))
    plenary_labels = np.ones(len(plenary_chunks))

    X = [x[1] for x in committee_chunks] + [x[1] for x in plenary_chunks]
    y = np.concatenate((committee_labels, plenary_labels))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    knn = KNeighborsClassifier()

    knn.fit(X_train, y_train)
    predicted_labels = knn.predict(X_test)
    print(classification_report(y_test, predicted_labels, target_names=['class committee', 'class plenary']), '\n')


def main(corpus_path, model_path):
    try:
        model = Word2Vec.load(os.path.join(model_path))
    except Exception as e:
        print(f"Couldn't load the model, make sure that the path is right. {e}")
        return
    tokenized_sentences, committee, plenary = get_excel(corpus_path)
    if tokenized_sentences is None or committee is None or plenary is None:
        return
    for chunk_size in [1, 3, 5]:
        try:
            classification_knn(model.wv, plenary, committee, chunk_size)
        except Exception as e:
            print(f"couldn't do classification for chunk_size = {chunk_size}, {e}")
            continue


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Please provide us the directory to the corpus file, and a path to model file.")
    else:
        main(sys.argv[1], sys.argv[2])


