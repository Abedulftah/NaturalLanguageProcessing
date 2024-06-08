import os

import numpy as np
import pandas as pd


class LM_Trigram:
    def __init__(self, corpus, model_type):
        self.corpus = corpus
        self.model_type = model_type
        self.unigrams_set = self.get_n_grams(1)
        self.bigrams_set = self.get_n_grams(2)
        self.trigrams_set = self.get_n_grams(3)
        self.length_of_corpus = 0
        for sentence in corpus:
            self.length_of_corpus += len(sentence.split())
        self.lambda_list = [0.7, 0.2, 0.1]

    def calculate_prob_of_sentence(self, sentence, smoothing):
        MLE = 0
        tokens = sentence.split()
        V = len(self.unigrams_set)
        if len(tokens) == 0:
            tokens.insert(0, '<s>')
            tokens.insert(0, "<s>")
        elif len(tokens) == 1:
            tokens.insert(0, "<s>")
        if smoothing == "Laplace":
            for i in range(len(tokens)):
                if i == 0:
                    V1, _, _ = self.duplications(tokens[i])
                    MLE = np.log2((V1 + 1) / (self.length_of_corpus + V))
                elif i == 1:
                    V1, V2, _ = self.duplications(tokens[i - 1], tokens[i])
                    MLE += np.log2((V2 + 1) / (V1 + V))
                else:
                    _, V2, V3 = self.duplications(tokens[i - 2], tokens[i - 1], tokens[i])
                    MLE += np.log2((V3 + 1) / (V2 + V))
        elif smoothing == "Linear":
            for i in range(len(tokens)):
                if i == 0:
                    V1, _, _ = self.duplications(tokens[i])
                    prob_2 = self.lambda_list[2] * ((V1 + 1) / (self.length_of_corpus + V))
                    MLE = np.log2(prob_2)
                elif i == 1:
                    V21, V2, _ = self.duplications(tokens[i - 1], tokens[i])
                    V1, _, _ = self.duplications(tokens[i])
                    if V21 == 0:
                        prob_2 = 0
                    else:
                        prob_2 = self.lambda_list[1] * (V2 / V21)
                    prob_1 = self.lambda_list[2] * ((V1 + 1) / (self.length_of_corpus + V))
                    MLE += np.log2(prob_2 + prob_1)
                else:
                    MLE += self.probability_of_tokens([tokens[i - 2], tokens[i - 1], tokens[i]])
        print(round(MLE, 3))
        return MLE

    def probability_of_tokens(self, tokens):
        _, V32, V3 = self.duplications(tokens[0], tokens[1], tokens[2])
        V21, V2, _ = self.duplications(tokens[1], tokens[2])
        V1, _, _ = self.duplications(tokens[2])
        if V32 == 0:
            prob_3 = 0
        else:
            prob_3 = self.lambda_list[0] * (V3 / V32)
        if V21 == 0:
            prob_2 = 0
        else:
            prob_2 = self.lambda_list[1] * (V2 / V21)
        prob_1 = self.lambda_list[2] * ((V1 + 1) / (self.length_of_corpus + len(self.unigrams_set)))
        return np.log2(prob_3 + prob_2 + prob_1)

    def generate_next_token(self, sentence):
        max_word, max_prob = None, 0
        tokens = sentence.split()
        for token in self.unigrams_set:
            if "".join(list(token)) == '<s>':
                continue
            if len(tokens) == 0:
                current_prob = self.probability_of_tokens(['<s>', '<s>', "".join(list(token))])
            elif len(tokens) == 1:
                current_prob = self.probability_of_tokens(['<s>', tokens[-1], "".join(list(token))])
            else:
                current_prob = self.probability_of_tokens([tokens[-2], tokens[-1], "".join(list(token))])
            if not max_word or current_prob > max_prob:
                max_word = token
                max_prob = current_prob
        print("".join(list(max_word)))
        return "".join(list(max_word))

    def duplications(self, token1, token2=None, token3=None):
        V1, V2, V3 = 0, 0, 0
        if tuple(token1) in self.unigrams_set:
            V1 = self.unigrams_set[tuple(token1)]
        if tuple([token1, token2]) in self.bigrams_set:
            V2 = self.bigrams_set[tuple([token1, token2])]
        if tuple([token1, token2, token3]) in self.trigrams_set:
            V3 = self.trigrams_set[tuple([token1, token2, token3])]

        return V1, V2, V3

    def get_n_grams(self, n):
        ngrams = {}
        for sentence in self.corpus:
            tokens = sentence.split()
            tokens.insert(0, "<s>")
            tokens.insert(0, "<s>")
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

    def get_k_n_collocations(self, k, n):
        collocations = []
        if n > 3:
            ngrams = self.get_n_grams(n)
        elif n == 3:
            ngrams = self.trigrams_set
        elif n == 2:
            ngrams = self.bigrams_set
        else:
            ngrams = self.unigrams_set

        length_ngrams = 0
        for key in ngrams:
            if '<s>' not in list(key):
                length_ngrams += ngrams[key]
        for sentence in self.corpus:
            tokens = sentence.split()
            for i in range(len(tokens) - n):
                pmi = self.get_PMI(tokens[i: i + n], ngrams, length_ngrams)
                collocations.append([tokens[i: i + n], pmi])
        collocations.sort(key=lambda x: x[1], reverse=True)
        return collocations[:k]

    def get_PMI(self, tokens, ngrams, length_ngrams):
        if tuple(tokens) not in ngrams:
            return 0
        length = len(tokens)
        one_gram_prob = 1
        for i in range(length):
            one_gram_prob *= self.unigrams_set[tuple(tokens[i])]
        return np.log2((ngrams[tuple(tokens)] * (self.length_of_corpus ** length)) / (one_gram_prob * length_ngrams))


def insert_collocations(models):
    with open('knesset_collocations.txt', 'w', encoding='utf-8') as file:
        for i, s in enumerate(['Two', 'Three', 'Four']):
            file.write(f"{s}-gram collocations:\n")
            for j in range(len(models)):
                try:
                    lst = models[j].get_k_n_collocations(10, i + 2)
                except Exception as e:
                    print(f"Couldn't get the collocations of {models[j].model_type}, we skipped, {e}")
                    continue
                file.write(f'{models[j].model_type} corpus:\n')
                for h in range(10):
                    file.write(" ".join(lst[h][0]) + "\n")
                file.write('\n')


def insert_sentence(models, line, sentences, list_generated_tokens, file):
    file.write(f"Original sentence: {line}")
    for i in range(len(models)):
        try:
            prob1 = models[(i + 1) % 2].calculate_prob_of_sentence(sentences[i], 'Linear')
            prob2 = models[i].calculate_prob_of_sentence(sentences[i], 'Linear')
        except Exception as e:
            prob1, prob2 = 0, 0
            print(f"Couldn't get the probability of the {sentences[i]}, {e}")
        # when need to check if the max between the 4 probabilities.
        if prob1 > prob2:
            highType = models[(i + 1) % 2].model_type
        else:
            highType = models[i].model_type

        file.write(f"{models[i].model_type} sentence: {sentences[i]}\n")
        file.write(f"{models[i].model_type} tokens: {','.join(list_generated_tokens[i])}\n")
        file.write(
            f"Probability of {models[i].model_type.lower()} sentence in {models[i].model_type.lower()} corpus: {'%.3f' % prob2}\n")
        file.write(
            f"Probability of {models[i].model_type.lower()} sentence in {models[(i + 1) % 2].model_type.lower()} corpus: {'%.3f' % prob1}\n")
        file.write(f"This sentence is more likely to appear in corpus: {highType.lower()}\n")
    file.write("\n")


def to_check():
    committee, plenary = get_excel('')
    if not committee or not plenary:
        return
    Cmodel = LM_Trigram(committee, "Committee")
    Pmodel = LM_Trigram(plenary, "Plenary")
    list_generated_tokensC = []
    list_generated_tokensP = []
    with open('sentences_result.txt', 'w', encoding='utf-8') as file1:
        with open('masked_sentences.txt', 'r', encoding='utf-8') as file:
            for line in file:
                lst_tokens = line.split()
                sentenceC = " "
                sentenceP = " "
                for token in lst_tokens:
                    if token == '[*]':
                        list_generated_tokensC.append(Cmodel.generate_next_token(sentenceC))
                        list_generated_tokensP.append(Pmodel.generate_next_token(sentenceP))
                        sentenceC += list_generated_tokensC[-1] + " "
                        sentenceP += list_generated_tokensP[-1] + " "
                    else:
                        sentenceC += token + " "
                        sentenceP += token + " "
                try:
                    insert_sentence([Cmodel, Pmodel], line, [sentenceC, sentenceP],
                                    [list_generated_tokensC, list_generated_tokensP], file1)
                except Exception as e:
                    print(f"Couldn't insert:\n {sentenceP}\n {sentenceC}\n into the file text {e}")
                list_generated_tokensP.clear()
                list_generated_tokensC.clear()
    try:
        insert_collocations([Cmodel, Pmodel])
    except Exception as e:
        print(f"Couldn't insert the collocations into a test file {e}")


def get_excel(input_path):
    # ptv committee, ptm plenary.
    try:
        if 'example_knesset_corpus.csv' in input_path:
            excel = pd.read_csv(input_path)
        else:
            excel = pd.read_csv(os.path.join(input_path, "example_knesset_corpus.csv"), encoding='utf-8')
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


to_check()
