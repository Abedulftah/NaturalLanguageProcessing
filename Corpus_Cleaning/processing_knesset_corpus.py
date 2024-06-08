import sys

from docx import Document
import pandas as pd
import os
import re


class Protocol:
    def __init__(self, name, number, typeMV, sentences):
        self._name = name
        self._number = number
        self._typeMV = typeMV
        self._sentences = sentences

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, value):
        self._number = value

    @property
    def typeMV(self):
        return self._typeMV

    @typeMV.setter
    def typeMV(self, value):
        self._typeMV = value

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        self._sentences = value


class Sentence:
    def __init__(self, name, sentences=None):
        if sentences is None:
            sentences = []
        self._name = name
        self._sentences = sentences

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        self._sentences = value


def get_name(name, typeMV):
    titles = {
        "שר ה", "סגן", "מזכיר", "ועדת", "נשיא", "ראש ה"
    }
    new_name = ''
    # ptv committee, ptm plenary.
    if typeMV == 'committee':
        minister = False
    else:
        minister = True
    if any(title in name for title in titles):
        minister = True
    elif '(' in name or ')' in name:
        minister = False
    h = name.split('<')
    final_lst = h.copy()
    for c in h:
        if '>' in c:
            final_lst = c.split('>')
            break
    for c in final_lst:
        if ':' not in c:
            final_lst.remove(c)
    while "" in final_lst:
        final_lst.remove("")
    if len(final_lst) < 1:
        return None
    name = final_lst[0].split(" ")
    for i in range(len(name)):
        if '(' in name[i] or ')' in name[i]:
            name = name[:i]
            break
    for i in range(len(name) - 1, -1, -1):
        if '"' in name[i] and not 'חרל"פ' in name[i]:
            break
        elif i + 3 == len(name) and minister:
            break
        else:
            if new_name == '':
                new_name = name[i] + new_name
            else:
                new_name = name[i] + ' ' + new_name
    return new_name.split(":")[0].replace("\t", " ").replace("\n", "")


def tokenization(sentence):
    # ' we don't need to change anything for it.
    sentence = sentence.split(" ")
    while "" in sentence:
        sentence.remove("")
    for mark in ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}']:
        for i in range(len(sentence)):
            sentence[i] = sentence[i].replace(f'{mark}', f' {mark} ')
    sentence[0] = " " + sentence[0]
    for i in range(len(sentence)):
        if sentence[i][-1] != ' ':
            sentence[i] = sentence[i] + " "
    sentence = "".join(sentence)
    sentence = sentence.replace(' "', ' " ')
    sentence = sentence.replace('" ', ' " ')
    sentence = sentence.replace('–\n', "")
    sentence = sentence.replace('\n–', "")
    sentence = sentence.replace('\t', " ")
    sentence = sentence.replace('\n', " ")
    return sentence


def boundary_founder(sentences):
    sentences_with_boundary = re.split(r'(?<=[.:;!?])\s+(?!\d\.\d)', sentences)
    lst = []
    for sentence in sentences_with_boundary:
        temp = clean_sentences(sentence)
        if temp is not None:
            lst.append(tokenization(temp))
    return lst


def clean_sentences(sentence):
    check_vlad = sentence.split(' ')
    if sentence == "":
        return None
    elif '---' in sentence.replace("\t", "").replace('\n',"").replace(" ", "") or '–––' in sentence.replace("\t", "").replace('\n',"").replace(" ", ""):
        return None
    elif len(check_vlad) < 4:
        return None
    elif not re.compile(r'[^@#%*\-–+/\n\t\s\"\',:;(){}\[\].!?0-9]').search(sentence):
        return None
    elif re.compile(r'[^#@%*\-–+/\n\t\s\"\',:;(){}\[\]א-ת.!?0-9]').search(sentence):
        return None
    return sentence


def is_name(text, style, bold, underline):
    s_words_nd = {
        "חברי הוועדה",
        "יועץ משפטי",
        "מוזמנים",
        "נכחו",
        "מנהלת הוועדה",
        "סדר היום",
        "יועצים משפטיים",
        "סדר-היום",
        "נוכחים",
        "קצרנית",
        "יועצת משפטית",
        "משתתפים (באמצעים מקוונים)",
        "הגילויים החדשים בנוגע לחפירות בהר הבית – הצעה לסדר של חברי הכנסת",
        "מנהל הוועדה",
        "חברי הכנסת",
        "מנהלי הוועדה",
        "האפשרות השנייה היא",
        "האפשרות השלישית היא",
        "משתתפים באמצעים מקוונים",
        "מוזמנים באמצעים מקוונים",
        "רשמת פרלמנטרית",
        "חברי כנסת",
        "רשמה וערכה",
        "מזכירת הוועדה",
        "קצרנית פרלמנטרית",
        "ייעוץ משפטי",
        "רישום פרלמנטרי",
        "מנהל/ת הוועדה"
    }

    if any(n in text for n in s_words_nd):
        return False
    last_char = text.replace("\t", "").replace(" ", "").replace("\n", "")[-1]
    if not bold and underline and (':' == last_char or '>' == last_char or '<' == last_char):
        return True
    elif bold and underline and (':' == last_char or '>' == last_char or '<' == last_char) \
            and ('Plain' in style or 'Head' in style):
        return True
    elif ':' == last_char and underline:
        return True
    return False


def to_excel(list_of_protocols, outputpath):
    columns = ['protocol_name', 'knesset_number', 'protocol_type', 'speaker_name', 'sentence_text']
    data = []

    for protocol in list_of_protocols:
        for sentence in protocol.sentences:
            if sentence.sentences and len(sentence.sentences) > 0:
                for s in sentence.sentences:
                    if s.replace("\t", " ").replace(" ", "").replace("\n", " ") != "":
                        data.append([protocol.name, protocol.number, protocol.typeMV, sentence.name, s])

    execl = pd.DataFrame(data, columns=columns)
    if 'knesset_corpus.csv' in outputpath:
        execl.to_csv(outputpath, index=False, encoding='utf-8')
    else:
        execl.to_csv(os.path.join(outputpath, 'knesset_corpus.csv'), index=False, encoding='utf-8')


def main(input_path, output_path, excluded_files=None):
    if excluded_files is None:
        excluded_files = []

    lstProtocols = []
    try:
        list_dir = os.listdir(input_path)
    except Exception as e:
        print(f'The path is wrong check if it has spaces you need to put it as a string, {e}')
        return
    for filename in list_dir:
        if filename in excluded_files:
            continue
        try:
            strs = filename.split('_')
            number = int(strs[0])
            if strs[1][-1] == 'm':
                typeMV = 'plenary'
            else:
                typeMV = 'committee'
        except Exception as e:
            print(f"Error: Unable to read the name of the file '{filename}'. Skipping this file.\n{e}")
            continue

        try:
            document = Document(os.path.join(input_path, filename))
            lstSentences = []
            sentencesNBound = ''

            for par in document.paragraphs:
                if not par.text.replace("\t", "").replace(" ", "").replace("\n", ""):
                    continue
                if par.style:
                    style_name = par.style.name
                else:
                    style_name = None
                bold, underline, speaker = False, False, style_name in ['דובר', 'אורח', 'יור', 'דובר-המשך', 'קריאות']
                if not speaker and len(par.runs) < 30:
                    for run in par.runs:
                        if bold and underline:
                            break
                        if run.bold:
                            bold = run.bold
                        if run.underline:
                            underline = run.underline
                if speaker or (len(par.runs) < 30 and is_name(par.text, style_name, bold, underline)):
                    if sentencesNBound != '':
                        lstSentences[-1].sentences += boundary_founder(sentencesNBound)
                    sentencesNBound = ''
                    name = get_name(par.text, typeMV)
                    if not name:
                        continue

                    lstSentences.append(Sentence(name))
                elif len(lstSentences) > 0:
                    sentencesNBound = sentencesNBound + " " + par.text

            if sentencesNBound != "" and len(lstSentences) > 0:
                lstSentences[-1].sentences += boundary_founder(sentencesNBound)
            lstProtocols.append(Protocol(filename, number, typeMV, lstSentences.copy()))
            lstSentences.clear()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    try:
        to_excel(lstProtocols, output_path)
    except Exception as e:
        print(f'Failed to load to CSV: {e}')


if __name__ == "__main__":
    excluded_files = ['18_ptm_175821.docx', '18_ptm_175868.docx']
    if len(sys.argv) < 3:
        print("Please provide us the directory to the word files, and where you want the output to be.")
    else:
        main(sys.argv[1], sys.argv[2], excluded_files)

