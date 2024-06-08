import os
import sys
from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
        return None, None
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
        return None
    return corpus_tokenized, committee, plenary


def similarity_between_words(model, save_output):
    with open(os.path.join(save_output, 'knesset_similar_words.txt'), 'w', encoding='utf-8') as knesset_similar_words:
        list_of_words = ['שולחן', 'שלום', 'חבר', 'ממשלה', 'כנסת', 'ישראל']
        for word in list_of_words:
            knesset_similar_words.write(f"{word}: ")
            to_insert = model.wv.most_similar(word, topn=5)
            for index in range(len(to_insert)):
                if index == 4:
                    knesset_similar_words.write(f'({to_insert[index][0]}, {str(to_insert[index][1])})\n')
                else:
                    knesset_similar_words.write(f'({to_insert[index][0]}, {str(to_insert[index][1])}), ')


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


def closest_sentences(embeddings, save_output):
    chosen_sentences_10 = ["לך אני אומר את זה .", "אבל עוד לא הגענו לשם .", "זאת נקודה מאוד מאוד חשובה .",
                      "יש כאן בעיה מאוד רחבה .", "אני רוצה לסיים במשפט אחרון .", "את מי זה משמש ?",
                      "אני מתחיל לקצוב את זמנך .", "אני מבקשת להפסיק את זה .", "אתה יודע על מה אני מדברת ?",
                      'על - פי תוכנית הפיתוח של היישובים הערביים , המשרד היה אמור להקצות 5 מיליוני ש"ח לדרכים חקלאיות ו- 10 מיליוני ש"ח למימון פרויקט בית - נטופה .']
    # new_word = ["רבותי חברי הכנסת , אני מסכם ומסיים ."]
    chosen_sentences = []
    for sentence in chosen_sentences_10:
        for i in range(len(embeddings)):
            if sentence == embeddings[i][0]:
                chosen_sentences.append(i)
    with open(os.path.join(save_output, 'knesset_similar_sentences.txt'), 'w', encoding='utf-8') as knesset_similar_sentences:
        for i in chosen_sentences:
            our_sentence = embeddings[i]
            best_score = 0
            best_index = -1
            for j in range(len(embeddings)):
                if j == i:
                    continue
                current_score = cosine_similarity(our_sentence[1].reshape(1, -1), embeddings[j][1].reshape(1, -1))
                if current_score > best_score:
                    best_score = current_score
                    best_index = j
            knesset_similar_sentences.write(f"{our_sentence[0]}: most similar sentence: {embeddings[best_index][0]}\n")


def replace_red_words(model, save_output):
    sentences = ['ברוכים הבאים , הכנסו בבקשה לחדר .',
                 'אני מוכנה להאריך את ההסכם באותם תנאים.',
                 'בוקר טוב, אני פותח את הישיבה .',
                 'שלום , הערב התבשרנו שחברינו היקר לא ימשיך איתנו בשנה הבאה .']
    old_sentences = sentences.copy()

    # this is for the first sentence.
    similar_word3 = model.wv.most_similar(positive=['בית'], topn=3)
    sentences[0] = sentences[0].replace('לחדר', similar_word3[1][0])

    # this is for the second sentence.
    similar_word1 = model.wv.most_similar(positive=['מוכנה', 'אישה'], negative=['גבר'], topn=3)
    similar_word2 = model.wv.most_similar(positive=['הסדר'],  topn=3)  # ההסכם אינה קיימת בקורפוס
    sentences[1] = sentences[1].replace('מוכנה', similar_word1[1][0])
    sentences[1] = sentences[1].replace('ההסכם', similar_word2[0][0])

    # this is for the third sentence.
    similar_word4 = model.wv.most_similar(positive=['מדהים'], topn=3)  # טוב
    similar_word5 = model.wv.most_similar(positive=['מתחיל'], topn=3)   # פותח
    sentences[2] = sentences[2].replace('טוב', similar_word4[0][0])
    sentences[2] = sentences[2].replace('פותח', similar_word5[0][0])

    # this is for the fourth sentence.
    similar_word6 = model.wv.most_similar(positive=['חברים'], topn=3)
    similar_word7 = model.wv.most_similar(positive=['הטוב'], topn=3)
    similar_word8 = model.wv.most_similar(positive=['בשנה'], topn=3)
    sentences[3] = sentences[3].replace('שלום', similar_word6[0][0])
    sentences[3] = sentences[3].replace('היקר', similar_word7[2][0])
    sentences[3] = sentences[3].replace('בשנה', similar_word8[1][0])

    with open(os.path.join(save_output, 'red_words_sentences.txt'), 'w', encoding='utf-8') as red_words_sentences:
        for i in range(len(sentences)):
            red_words_sentences.write(f"{old_sentences[i]}: {sentences[i]}\n")


def main(corpus_input, save_output):
    tokenized_sentences, committee, plenary = get_excel(corpus_input)
    print(tokenized_sentences)
    if tokenized_sentences is None or committee is None or plenary is None:
        return
    try:
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
    except Exception as e:
        print(f"Couldn't make a model, {e}.")
        return
    try:
        model.save(os.path.join(save_output, 'knesset_word2vec.model'))
    except Exception as e:
        print(f"Couldn't save the model, {e}.")
        return

    try:
        similarity_between_words(model, save_output)
    except Exception as e:
        print(f"Couldn't find the similarity between the given words and save them, {e}.")

    try:
        embeddings = sentence_embeddings(model.wv, committee + plenary)
    except Exception as e:
        print(f"Couldn't make the embeddings vectors, {e}.")
        return

    try:
        closest_sentences(embeddings, save_output)
    except Exception as e:
        print(f"Couldn't choose 10 sentences, find the closest to them and save, {e}.")

    try:
        replace_red_words(model, save_output)
    except Exception as e:
        print(f"Couldn't replace the red words and save, {e}.")


if __name__ == '__main__':
    tokenized_sentences, committee, plenary = get_excel('example_knesset_corpus.csv')
    print(tokenized_sentences)
    if len(sys.argv) < 3:
        print("Please provide us the directory to the corpus file, and a path to save the output files.")
    else:
        main(sys.argv[1], sys.argv[2])
