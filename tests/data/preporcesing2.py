from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_tweets(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return [tweet.strip() for tweet in tweets]

def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        labels = file.readlines()
    return [label.strip() for label in labels]

tweets_file_path = 'train_text.txt'
emojis_file_path = 'train_labels.txt'

# Load tweets and labels
X = load_tweets(tweets_file_path)
y = load_labels(emojis_file_path)

# Assuming 'X' contains your preprocessed tweet text and 'y' contains corresponding emojis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")