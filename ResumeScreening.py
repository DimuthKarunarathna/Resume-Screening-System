import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
import joblib
import PyPDF2
import re
import string
import streamlit as st
import base64

# Load the model, vectorizer, and label encoder
model = joblib.load("models/resume_classifier.pkl")
tfidf = joblib.load("models/vectorizer.pkl")
le = joblib.load("models/label_encoder.pkl")


# Define stop words (ensure this is consistent with your training)
stop_words = set(stopwords.words('english'))

# Define functions (clean_text, extract_text_from_pdf)
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove links
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Lowercase
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return ' '.join(words)


# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image:
#         encoded = base64.b64encode(image.read()).decode()
#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{encoded}");
#         background-size: cover;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# # Call this function before your app UI starts
# add_bg_from_local('background.jpg')


# Streamlit app title
st.title("Resume Screening System")

# File uploader
uploaded_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Extract text
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text:
        # Clean text
        cleaned_resume_text = clean_text(resume_text)

        # Vectorize text
        vectorized_resume = tfidf.transform([cleaned_resume_text])

        # Make prediction
        predicted_label = model.predict(vectorized_resume)

        # Decode prediction
        predicted_category = le.inverse_transform(predicted_label)

        # Display result
        st.subheader("Predicted Category:")
        st.write(predicted_category[0])


