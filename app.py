import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# List of common skills to track
SKILLS_DB = ["Python", "Java", "SQL", "Machine Learning", "Data Analysis", "Communication", "Project Management", "React", "AWS", "Excel"]

def extract_text(file):
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages]).lower()

st.set_page_config(page_title="AI Resume Screener Pro", layout="wide")
st.title("🤖 AI-Powered Resume Screening System")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    jd = st.text_area("Paste requirements here...", height=200).lower()

with col2:
    st.subheader("Upload Resumes")
    resumes = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze & Rank"):
    if jd and resumes:
        resume_texts = [extract_text(r) for r in resumes]
        
        # AI Match Score
        data = [jd] + resume_texts
        cv = TfidfVectorizer()
        matrix = cv.fit_transform(data)
        scores = cosine_similarity(matrix[0:1], matrix[1:])
        
        st.write("### Ranking Results:")
        for i, score in enumerate(scores[0]):
            st.divider()
            st.info(f"**{resumes[i].name}**: {score*100:.2f}% Match Score")
            
            # Missing Skills Logic
            missing = [skill for skill in SKILLS_DB if skill.lower() in jd and skill.lower() not in resume_texts[i]]
            if missing:
                st.warning(f"⚠️ **Missing Skills:** {', '.join(missing)}")
            else:
                st.success("✅ All key skills found!")
    else:
        st.warning("Please provide both a job description and resumes.")
