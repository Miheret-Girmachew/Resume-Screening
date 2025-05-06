import streamlit as st
import pandas as pd
import re, string
import nltk
import PyPDF2

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

nltk.download('stopwords')
nltk.download('punkt')


def clean_resume(text):
    """Clean resume text by removing URLs, punctuation, and non-ASCII characters."""
    text = re.sub(r'http\S+', ' ', text)  
    text = re.sub(r'RT|cc', ' ', text)    
    text = re.sub(r'#\S+', '', text)      
    text = re.sub(r'@\S+', ' ', text)     
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text) 
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    text = re.sub(r'\s+', ' ', text)     
    return text.strip()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyPDF2."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

@st.cache_data
def load_training_data():
    """Load and return the training dataset of resumes."""
    try:
        df = pd.read_csv('resume_dataset.csv', encoding='utf-8')
        return df
    except FileNotFoundError:
        st.error("The training dataset 'resume_dataset.csv' was not found.")
        return pd.DataFrame(columns=['Resume', 'Category'])

@st.cache_resource
def train_model(df):
    """Train the TF-IDF vectorizer and classifier on the resume dataset."""
    if df.empty:
        return None, None, None

    df['cleaned_resume'] = df['Resume'].apply(clean_resume)

    le = LabelEncoder()
    df['Category_encoded'] = le.fit_transform(df['Category'])

    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
    vectorizer.fit(df['cleaned_resume'])
    X = vectorizer.transform(df['cleaned_resume'])
    y = df['Category_encoded']

    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X, y)

    return vectorizer, clf, le

def predict_candidate_resume(resume_text, vectorizer, clf, le):
    """Clean and vectorize a resume text, then predict its category."""
    cleaned_text = clean_resume(resume_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)
    predicted_category = le.inverse_transform(prediction)
    return predicted_category[0]

st.title("Resume Screening Application (PDF Upload)")

st.sidebar.header("Job Posting Details")
job_category = st.sidebar.selectbox(
    "Select the Job Category",
    options=[
        "Data Science", "HR", "Advocate", "Arts", "Web Designing",
        "Mechanical Engineer", "Sales", "Health and fitness", "Civil Engineer",
        "Java Developer", "Business Analyst", "SAP Developer", "Automation Testing",
        "Electrical Engineering", "Operations Manager", "Python Developer",
        "DevOps Engineer", "Network Security Engineer", "PMO", "Database", "Hadoop",
        "ETL Developer", "DotNet Developer", "Blockchain", "Testing"
    ]
)
st.sidebar.write("Selected Job Category:", job_category)

st.header("Candidate Resume Screening")
st.write("Upload a PDF file containing the candidate resume.")

upload_option = st.radio("How would you like to provide candidate resumes?", 
                          ("Upload PDF File", "Enter Text"))

training_data = load_training_data()
if not training_data.empty:
    vectorizer, clf, le = train_model(training_data)
else:
    vectorizer, clf, le = None, None, None


if upload_option == "Upload PDF File":
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_pdf is not None:
        candidate_resume_text = extract_text_from_pdf(uploaded_pdf)
        st.write("Extracted resume text:")
        st.write(candidate_resume_text[:500] + "...")
        if vectorizer and clf and le:
            predicted_cat = predict_candidate_resume(candidate_resume_text, vectorizer, clf, le)
            st.write("**Predicted Category:**", predicted_cat)
            if predicted_cat == job_category:
                st.success("Candidate matches the job posting!")
            else:
                st.error("Candidate does not match the job posting.")
        else:
            st.error("Model is not trained. Please ensure the training dataset is available.")


else:
    candidate_resume_text = st.text_area("Enter Candidate Resume Text")
    if st.button("Screen Candidate"):
        if candidate_resume_text:
            if vectorizer and clf and le:
                predicted_cat = predict_candidate_resume(candidate_resume_text, vectorizer, clf, le)
                st.write("**Predicted Category:**", predicted_cat)
                if predicted_cat == job_category:
                    st.success("Candidate matches the job posting!")
                else:
                    st.error("Candidate does not match the job posting.")
            else:
                st.error("Model is not trained. Please ensure the training dataset is available.")
        else:
            st.warning("Please enter a resume text.")

st.write("**Note:** This demo uses a simple ML pipeline to classify resumes. In production, you might further tune the model and enhance the UI.")


# source venv/Scripts/activate
# pip install -r requirements.txt
# python data_analysis.py
# streamlit run app.py