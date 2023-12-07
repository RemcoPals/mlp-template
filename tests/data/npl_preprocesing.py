import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_en(word):
    # removes words in between hashtags
    word = re.sub(r'##AT##', ' ', word)

    # Remove non-English/non-German letters, punctuation (except '), numbers, and other characters
    word = re.sub(r'<.*?>|\(.*?\)', '', word)
    word = re.sub(r'[^a-zA-ZÄÖÜäöüß)(\s]', '', word)

    # removes html tags
    # word = re.sub(r"[^a-zA-Z'’<>]+|<[^a-z>]+>", ' ', word)

    # Remove repeating letters more than 5 times
    word = re.sub(r"(.)\1{4,}", r"\1\1\1\1\1", word)

    word = word.lower()
    return word

def text_to_dataframe(filename):
    # Read the text file
    with open(filename, 'r', encoding="utf8") as file:
        lines = file.readlines()

    # Split each line by tab and create a list of lists
    data = [line.strip().split('\n') for line in lines]
    df = pd.DataFrame(data)
    return df

#initialize filepaths
training_file = 'train_text.txt'

#create dataframe representations
training = text_to_dataframe(training_file)
#check length
print("length dataset :", len(training))

en_training=training.reset_index().drop("index",axis=1)
data = pd.DataFrame()
data["en"]=training

data["en"]=[clean_en(i).split() for i in data["en"]]

print("Dataset length : \n English dataset : ",len(data["en"]))
import statistics as stat
en=[len(i)for i in data["en"]]
n=50
en_max=0
for i in range(len(en)):
    if en[i]>n:
        en_max+=1
en_vocab = max(en)

print("Max sentence length,\n in English : {} ".format(max(en)))
print("Mean sentence length,\n in English : {} ".format(stat.mean(en)))
print(f"Num of sentences with more than {n} sentences\n in English : {en_max}")

X_train=data["en"]
print(type(X_train))

vektorizer = TfidfVectorizer(max_features=1000)
x = vektorizer.fit_transform(X_train)
print(x)
