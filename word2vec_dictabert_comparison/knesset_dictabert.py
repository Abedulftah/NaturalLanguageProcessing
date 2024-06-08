import os
import re
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
model.eval()


def fill_masked(masked_sentences_path, output_path):
    with open(os.path.join(output_path, 'dictabert_results.txt'), 'w', encoding='utf-8') as file1:
        with open(os.path.join(masked_sentences_path), 'r', encoding='utf-8') as file:
            for line in file:
                masked_text = re.sub(r'\[\*\]', "[MASK]", line)
                masked_list = masked_text.split()

                input_ids = tokenizer.encode(masked_text, return_tensors='pt')
                mask_positions = np.where(input_ids == tokenizer.mask_token_id)
                output = model(input_ids)
                top_k_logits = output.logits[mask_positions].squeeze(1)
                top_k_indices = torch.topk(top_k_logits, 1, dim=-1)[1]
                list_of_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.squeeze().tolist())
                if type(list_of_tokens) == str:
                    list_of_tokens = [list_of_tokens]
                j = 0
                for i in range(len(masked_list)):
                    if masked_list[i] == '[MASK]':
                        masked_list[i] = list_of_tokens[j]
                        j += 1
                file1.write(f'Original sentence: {line}')
                file1.write(f'DictaBERT sentence: {" ".join(masked_list)}\n')
                file1.write(f'DictaBERT tokens: {",".join(list_of_tokens)}\n\n')


def main(masked_sentences_path, output_path):
    try:
        fill_masked(masked_sentences_path, output_path)
    except Exception as e:
        print(f"Couldn't continue to save and replace the red words, {e}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Please provide us the directory to the masked_sentences.txt, and a path to save the output files.")
    else:
        main(sys.argv[1], sys.argv[2])