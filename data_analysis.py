import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import string
import nltk

# Suppress warnings
warnings.filterwarnings('ignore')

# Import scikit-learn modules
from sklearn.naive_bayes import MultinomialNB  # (Not used in this example but available)
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Additional libraries for visualization
from pandas.plotting import scatter_matrix
from matplotlib.gridspec import GridSpec
import seaborn as sns
from wordcloud import WordCloud
from scipy.sparse import hstack

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# -----------------------------
# Load Data and Initial Checks
# -----------------------------
resumeDataSet = pd.read_csv('resume_dataset.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''

print("First few rows of dataset:")
print(resumeDataSet.head())

print("Displaying the distinct categories of resume -")
print(resumeDataSet['Category'].unique())

print("Displaying the distinct categories of resume and the number of records belonging to each category -")
print(resumeDataSet['Category'].value_counts())

# -----------------------------
# Visualization: Category Counts
# -----------------------------
plt.figure(figsize=(15, 15))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=resumeDataSet)
plt.title("Resume Category Counts")
plt.show()

# -----------------------------
# Visualization: Category Distribution Pie Chart
# -----------------------------
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels = resumeDataSet['Category'].unique()

plt.figure(figsize=(25, 25))
the_grid = GridSpec(2, 2)
cmap = plt.get_cmap('coolwarm')
# Adjust number of colors to number of categories
colors = [cmap(i) for i in np.linspace(0, 1, len(targetCounts))]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()

# -----------------------------
# Define Resume Cleaning Function
# -----------------------------
def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)         # remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)            # remove hashtags
    resumeText = re.sub(r'@\S+', ' ', resumeText)           # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)   # remove non-ASCII characters
    resumeText = re.sub(r'\s+', ' ', resumeText)            # remove extra whitespace
    return resumeText.strip()

# Apply cleaning to the resume column
resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))

# -----------------------------
# Text Processing, WordCloud, and Frequency Distribution
# -----------------------------
oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
totalWords = []
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""

# Process the first 160 resumes (or as many as available)
for i in range(min(160, len(Sentences))):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += " " + cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word.lower() not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print("Most common words:", mostcommon)

# Generate and display word cloud
wc = WordCloud(width=800, height=400, background_color='white').generate(cleanedSentences)
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Resume Texts")
plt.show()

# -----------------------------
# Encode Categories
# -----------------------------
le = LabelEncoder()
resumeDataSet['Category_encoded'] = le.fit_transform(resumeDataSet['Category'])

# -----------------------------
# Feature Extraction and Model Training
# -----------------------------
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category_encoded'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print("Feature extraction completed ...")

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
print("\nClassification report for classifier:\n")
print(metrics.classification_report(y_test, prediction))
