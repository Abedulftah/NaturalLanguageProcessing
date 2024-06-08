import math
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main(input_path, output_path):
    d = {}
    try:
        if 'knesset_corpus.csv' in input_path:
                execl = pd.read_csv(input_path)
        else:
            execl = pd.read_csv(os.path.join(input_path, "knesset_corpus.csv"))
    except Exception as e:
        print(f"Couldn't read the csv file, {e}")
        return
    try:
        for value in execl['sentence_text']:
            tokens = value.split(" ")
            for token in tokens:
                if token in d:
                    d[token] += 1
                # "'," it is because we read the array as a string.
                elif token not in ['\'', '?', '!', '.', ',', '\"', '(', ")", '{', '}', "['", "']", "',", "", '-', '–']:
                    d[token] = 1
        lst = list(d.items())
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        print(len(lst))
        print("top 10: ")
        for i in range(10):
            print(lst[i])
        print("lowest 10: ")
        for i in range(1, 11):
            print(lst[-i])
        ranks = []
        frequencies = []

        for i in range(len(lst)):
            ranks.append(math.log2(i + 1))
            frequencies.append(math.log2(lst[i][1]))
    except Exception as e:
        print(f'An error occurred while reading the csv, make sure that the csv file is made as in the instruction, {e}')
        return
    try:
        plt.figure()
        plt.plot(ranks, frequencies, marker='', linestyle='-', color='blue')
        plt.title("Zipf's graph")
        plt.xlabel('log Rank')
        plt.ylabel('log Frequency')
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        if os.path.splitext(output_path)[1].lower() in extensions:
            plt.savefig(output_path)
        else:
            plt.savefig(os.path.join(output_path, "Zipf graph.png"))
    except Exception as e:
        print(
            f'An error occurred while trying to make an image, {e}')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide us the directory to the csv file, and where you want the output to be.")
    else:
        main(sys.argv[1], sys.argv[2])