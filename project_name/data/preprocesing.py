from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Preprocesing:
    tweets_file_path = 'data/train_text.txt'
    emojis_file_path = 'data/train_labels.txt'
    X_train=0
    X_test=0
    X_test_locked=0
    y_train=0
    y_test=0
    y_test_locked=0
    X_train_tfidf=0
    X_test_locked_tfidf=0
    X_test_tfidf=0


    # parameterized constructor
    def __init__(self):
        X = self.load_tweets()
        y = self.load_labels()
        self.X_train, self.X_test, self.X_test_locked, self.y_train, self.y_test, self.y_test_locked = self.split_data(X,y)
        self.X_train_tfidf, self.X_test_locked_tfidf, self.X_test_tfidf =self.to_tfidf()


    def load_tweets(self):
        with open(self.tweets_file_path, 'r', encoding='utf-8') as file:
            tweets = file.readlines()
        return [tweet.strip() for tweet in tweets]

    def load_labels(self):
        with open(self.emojis_file_path, 'r', encoding='utf-8') as file:
            labels = file.readlines()
        return [label.strip() for label in labels]

    def split_data(self,X,y):
        # Assuming 'X' contains your preprocessed tweet text and 'y' contains corresponding emojis
        X_train, X_test_locked, y_train, y_test_locked = train_test_split(X, y, test_size=0.2, stratify=y,
                                                                          random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train,
                                                            random_state=42)
        return X_train, X_test,X_test_locked, y_train, y_test,y_test_locked


    def to_tfidf(self):
        # Convert text data to numerical features using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', max_features=5000, sublinear_tf=True)
        X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train)
        X_test_locked_tfidf = tfidf_vectorizer.transform(self.X_test_locked)
        X_test_tfidf = tfidf_vectorizer.transform(self.X_test)
        return X_train_tfidf,X_test_locked_tfidf,X_test_tfidf
