import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
file_path = r"C:\Users\Lenovo\Downloads\Week-12\Week-12\IMDB Dataset.csv\IMDB Dataset.csv"
df = pd.read_csv(file_path)

# Step 1: Data Preprocessing

# Convert sentiment labels to binary (1 for positive, 0 for negative)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Step 2: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 3: Modeling

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)

# Step 4: Evaluation
nb_report = classification_report(y_test, nb_pred, output_dict=True)
lr_report = classification_report(y_test, lr_pred, output_dict=True)
svm_report = classification_report(y_test, svm_pred, output_dict=True)

# Print accuracy of each model
print("Naive Bayes Accuracy:", nb_report['accuracy'])
print("Logistic Regression Accuracy:", lr_report['accuracy'])
print("SVM Accuracy:", svm_report['accuracy'])

# Print the detailed classification report for each model
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, nb_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

# Predict the number of positive and negative reviews for each model
def count_predictions(predictions):
    return pd.Series(predictions).value_counts()

nb_counts = count_predictions(nb_pred)
lr_counts = count_predictions(lr_pred)
svm_counts = count_predictions(svm_pred)

print("\nNaive Bayes Predictions Count:\n", nb_counts)
print("\nLogistic Regression Predictions Count:\n", lr_counts)
print("\nSVM Predictions Count:\n", svm_counts)
