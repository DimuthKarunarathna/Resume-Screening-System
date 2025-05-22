import streamlit as st
import joblib
import PyPDF2  # For extracting text from PDF files

# Load model and vectorizer
model = joblib.load("/Users/dimuthshiharakarunarathna/Desktop/kaggle/Resume/resume_classifier.pkl")
vectorizer = joblib.load("/Users/dimuthshiharakarunarathna/Desktop/kaggle/Resume/vectorizer.pkl")

# App UI
st.title("Resume Screening System")
st.write("Upload a PDF resume or paste resume text to classify its job category")

# Job category mapping
job_categories = {
    0: "Software Engineer",
    1: "Data Scientist",
    2: "Project Manager",
    3: "Marketing Specialist",
    4: "Sales Representative",
    5: "Graphic Designer",
    6: "HR Specialist",
    7: "Business Analyst",
    8: "Content Writer",
    9: "Financial Analyst",
    10: "Mechanical Engineer",
    11: "Civil Engineer",
    12: "Electrical Engineer",
    13: "Teacher",
    14: "Nurse",
    15: "Other"  # Add more categories as needed
}

# File uploader
uploaded_file = st.file_uploader("Upload a PDF resume", type=["pdf"])

# Text input
resume_text = ""
if uploaded_file is not None:
    try:
        # Read and extract text from the uploaded PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        st.success("PDF uploaded and text extracted successfully!")
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
else:
    resume_text = st.text_area("Or paste resume text here", height=300)

if st.button("Classify"):
    if resume_text.strip() == "":
        st.warning("Please provide resume text either by uploading a PDF or pasting it.")
    else:
        # Vectorize and predict
        vectorized = vectorizer.transform([resume_text])
        prediction = model.predict(vectorized)
        job_category = job_categories.get(prediction[0], "Unknown Category")
        st.success(f"Predicted Job Category: **{job_category}**")