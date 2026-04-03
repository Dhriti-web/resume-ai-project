import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#helper function to read PDFs
def extract_text(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages])

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI-Powered Resume Screening System")

# Two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    jd = st.text_area("Paste requirements here...", height=200)

with col2:
    st.subheader("Upload Resumes")
    resumes = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank"):
    if jd and resumes:
        resume_texts = [extract_text(r) for r in resumes]
        
        # AI Logic: Converting text to numbers (Vectors) and comparing them
        data = [jd] + resume_texts
        cv = TfidfVectorizer()
        matrix = cv.fit_transform(data)
        vectors = matrix.toarray()
        
        # Compare the JD (first item) with all resumes
        scores = cosine_similarity([vectors[0]], vectors[1:])[0]
        
        # Display results
        st.write("### Ranking Results:")
        for i, score in enumerate(scores):
            st.info(f"*{resumes[i].name}*: {score*100:.2f}% Match Score")
    else:
        st.warning("Please provide both a job description and resumes.")