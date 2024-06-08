import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
import pandas as pd
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
CHUNK_SIZE = 3
random.seed(42)
WHICH = 'OUR'


def get_n_grams(chunk, n):
    ngrams = {}
    for sentence in chunk:
        tokens = sentence.split()
        for i in range(len(tokens) - n + 1):
            if n > 1:
                tuple_tokens = tuple(tokens[i: i + n])
            else:
                tuple_tokens = tuple(tokens[i])
            if tuple_tokens in ngrams:
                ngrams[tuple_tokens] += 1
            else:
                ngrams[tuple_tokens] = 1
    return ngrams


def get_excel(input_path):
    # ptv committee, ptm plenary.
    try:
        if 'example_knesset_corpus.csv' in input_path:
            excel = pd.read_csv(input_path)
        else:
            excel = pd.read_csv(os.path.join(input_path, "../example_knesset_corpus.csv"), encoding='utf-8')
    except Exception as e:
        print(f"Couldn't read the csv file, {e}")
        return None, None
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
    except Exception as e:
        print(f"Couldn't iterate over the csv file, {e}")
        return None, None
    return committee, plenary


def make_chunks():
    committee, plenary = get_excel("")
    print('committee sentences: ', len(committee), 'plenary sentences: ', len(plenary))
    committee_chunks, plenary_chunks, chunk = [], [], []
    # we took every 5 sentences to make it as a chunk for committee.
    for i in range(len(committee)):
        chunk.append(committee[i])
        if (i + 1) % CHUNK_SIZE == 0:
            committee_chunks.append(" ".join(chunk.copy()))
            chunk.clear()

    chunk.clear()
    # we took every 5 sentences to make it as a chunk for plenary.
    for i in range(len(plenary)):
        chunk.append(plenary[i])
        if (i + 1) % CHUNK_SIZE == 0:
            plenary_chunks.append(" ".join(chunk.copy()))
            chunk.clear()

    return committee_chunks, plenary_chunks


def down_sampling(c_chunks, p_chunks):
    print('committee: ', len(c_chunks), 'plenary: ', len(p_chunks))
    if len(c_chunks) < len(p_chunks):
        return c_chunks, random.sample(p_chunks, len(c_chunks))
    return random.sample(c_chunks, len(p_chunks)), p_chunks


def create_feature_vector(co_chunks, pl_chunks):
    tfidf = TfidfVectorizer()
    tfidf.fit(co_chunks + pl_chunks)
    features = tfidf.get_feature_names_out()
    committee_unigram = get_n_grams(co_chunks, 1)
    plenary_unigram = get_n_grams(pl_chunks, 1)
    lst = []
    for feature in features:
        lst.append([feature, abs(committee_unigram.get(tuple(feature), 0) - plenary_unigram.get(tuple(feature), 0))])
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = [sub[0] for sub in lst]
    lst = lst[:2000]
    our_tfidf = TfidfVectorizer(vocabulary=lst)
    our_BoW = our_tfidf.fit_transform(co_chunks + pl_chunks)
    return tfidf, our_tfidf, our_BoW


def train_test(vectorBoW, length):
    committee_labels = np.zeros(length)
    plenary_labels = np.ones(length)

    X = vectorBoW
    y = np.concatenate((committee_labels, plenary_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    svm_model = svm.SVC(kernel='linear')
    knn = KNeighborsClassifier()

    knn.fit(X_train, y_train)
    predicted_labels = knn.predict(X_test)
    accuracyknn = np.mean(predicted_labels == y_test)
    print(f'{WHICH}_10_90_KNN classification report:')
    print(classification_report(y_test, predicted_labels, target_names=['class committee', 'class plenary']), '\n')

    svm_model.fit(X_train, y_train)
    predicted_labels = svm_model.predict(X_test)
    accuracysvm = np.mean(predicted_labels == y_test)
    print(f'{WHICH}_10_90_SVM classification report:')
    print(classification_report(y_test, predicted_labels, target_names=['class committee', 'class plenary']), '\n')
    if accuracysvm > accuracyknn:
        return svm_model, accuracysvm
    return knn, accuracyknn


def k_fold(vectorBoW, length):
    svm_model = svm.SVC(kernel='linear')
    knn = KNeighborsClassifier()
    committee_labels = np.zeros(length)
    plenary_labels = np.ones(length)

    X = vectorBoW
    y = np.concatenate((committee_labels, plenary_labels))
    ten_folds = StratifiedKFold(n_splits=10, shuffle=True)
    knn_labels = cross_val_predict(knn, X, y, cv=ten_folds, n_jobs=-1)
    print(f'{WHICH}_10Fold_KNN classification report:')
    print(classification_report(y, knn_labels, target_names=['class committee', 'class plenary']), '\n')
    knn_acc = np.mean(y == knn_labels)
    svm_labels = cross_val_predict(svm_model, X, y, cv=ten_folds, n_jobs=-1)
    print(f'{WHICH}_10Fold_SVM classification report:')
    print(classification_report(y, svm_labels, target_names=['class committee', 'class plenary']), '\n')
    svm_acc = np.mean(y == svm_labels)


def classification(classify_model, classify_tfidf):
    chunks, predicted = [], []
    with open('knesset_text_chunks.txt', 'r', encoding='utf-8') as knesset_chunks:
        for line in knesset_chunks:
            chunks.append(line.strip())
    vector = classify_tfidf.transform(chunks)
    predicted = classify_model.predict(vector)
    with open('classification_result.txt', 'w', encoding='utf-8') as output:
        for label in predicted:
            if label == 0:
                output.write('committee\n')
            else:
                output.write('plenary\n')


def main():
    try:
        committee_chunks, plenary_chunks = make_chunks()
    except Exception as e:
        print(f'We couldn\'t read, divide the sentences into chunks, {e}.')
        return
    try:
        committee_chunks, plenary_chunks = down_sampling(committee_chunks, plenary_chunks)
    except Exception as e:
        print(f'We couldn\'t down sampling to the chunking, {e}.')
        return
    try:
        tfidf, our_tfidf, our_BoW = create_feature_vector(committee_chunks, plenary_chunks)
    except Exception as e:
        print(f'We could\'t create the feature vectors, {e}.')
        return
    print('committee: ', len(committee_chunks), 'plenary: ', len(plenary_chunks))
    try:
        BoW = tfidf.transform(committee_chunks + plenary_chunks)
        print("vectorBoW shape: ", BoW.shape)
    except Exception as e:
        print(f'We couldn\'t do BoW for every chunk, {e}.')
    print("ourBoW shape: ", our_BoW.shape)

    # try:
    #     our_model, our_acc = train_test(our_BoW, len(committee_chunks))
    # except Exception as e:
    #     print(f'We couldn\'t do train_test using our BoW, {e}.')
    WHICH = ''
    try:
        model, acc = train_test(BoW, len(plenary_chunks))
    except Exception as e:
        print(f'We couldn\'t do train_test using BoW for question 1, {e}.')
    WHICH = 'our'
    try:
        k_fold(our_BoW, len(committee_chunks))
    except Exception as e:
        print(f'We couldn\'t do k_fold using our_BoW, {e}.')
    WHICH = ''
    try:
        k_fold(BoW, len(committee_chunks))
    except Exception as e:
        print(f'We couldn\'t do k_fold using BoW from question 1, {e}.')

    try:
        classification(model, tfidf)
    except Exception as e:
        print(f'We couldn\'t do classification and we didn\'t do the classification_result.txt, {e}.')


if __name__ == "__main__":
    main()