import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


file_path = 'train_text.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    corpus =  [line.strip() for line in file]

stop_words=set(stopwords.words('English'))
lemmatizer = WordNetLemmatizer()
vektorizer = TfidfVectorizer(max_features=1000)

def preproces(example_text):
    example_text =example_text.lower()
    example_text = word_tokenize(example_text)
    example_text= [word for word in example_text if word.isalnum()]
    example_text=[word for word in example_text if word not in stop_words]
    example_text = [lemmatizer.lemmatize(word) for word in example_text]
    return example_text

listt=[]
for i in range(len(corpus)):
    listt.append(preproces(corpus[i]))

x = vektorizer.fit_transform(listt)
