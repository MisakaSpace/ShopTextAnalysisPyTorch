import pandas as pd
import re


def process(text: str):
    text = text.lower()
    text = re.sub(r'@\s*[\w\d]*', '', text)

    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub(r'&quot', " ", text)
    text = re.sub(r'&lt', " ", text)
    text = re.sub(r'&#39', "'", text)
    text = re.sub(r'rt', ' ', text)
    text = re.sub(r'd', ' ', text)
    text = re.sub(r'[\?:\)\(\/\\,;%:\*_!\-=\.]', ' ', text)
    text = re.sub(r'\#\s*[\w\d]*', '', text)
    text = text.replace('"', "")
    while True:
        if "  " not in text:
            break
        text = text.replace("  ", " ")
    while True:
        if "\n" not in text:
            break
        text = text.replace("\n", "")
    text = text.replace("' ", "'")
    return text.strip()

if __name__ == '__main__':
    n = ['original', 'translate']
    data_positive = pd.read_csv('data/positive.txt', sep=',', error_bad_lines=False, names=n)
    data_negative = pd.read_csv('data/negative.txt', sep=',', error_bad_lines=False, names=n)

    with open('data/positive_clear.txt', 'w', encoding='utf-8') as file:
        for i in data_positive['translate']:
            processed = process(str(i))
            if processed and processed != 'nan':
                file.write('"{}"\n'.format(processed))

    with open('data/negative_clear.txt', 'w', encoding='utf-8') as file:
        for i in data_negative['translate']:
            processed = process(str(i))
            if processed != 'nan':
                file.write('"{}"\n'.format(processed))
