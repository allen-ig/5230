import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer


def load_and_process_data(filename, stopwords_list):

    rtitle = re.compile(r'^#\*\s*(.+)')
    stemmer = SnowballStemmer('english')
    titles = []
    stopwords = []
    parsed_titles = []

    with open(filename) as fin:
        for line in fin.readlines():
            line = line.strip('\n')
            mtitle = rtitle.match(line)
            ## if it is a title string
            if mtitle:
                title = mtitle.group(1).lower()
                titles.append(title[:-1])
    fin.close()
    print("===== raw dataset loaded =====")

    with open(stopwords_list) as fsw:
        for word in fsw.readlines():
            word = word.strip('\n')
            stopwords.append(word)
    fsw.close()
    print("===== stopwords list loaded =====")
    print("===== start to parse text =====")
    fout = open('preprocessed.txt', 'w+')
    for i, words in enumerate(titles):
        words = words.split(" ")
        words = [re.match('[a-zA-Z0-9]+', stemmer.stem(word)).group() for word in words if re.match('[a-zA-Z0-9]+', stemmer.stem(word)) is not None]
        words = ['NUM' if re.match('[0-9]+', word) is not None else word for word in words]
        words = list(filter(None, ["" if word in stopwords else word for word in words]))
        words = ' '.join(words)
        fout.write(words + '\n')
        parsed_titles.append(words)
        if i % 100000 == 0 :
            print("===== %d titles are parsed successfully =====" % i)
    fout.close()
    print("===== parsing completed. parsed data was saved in file preprocessed.txt, please load this as the input for your model. End. =====")

if __name__ == '__main__':
    main()
