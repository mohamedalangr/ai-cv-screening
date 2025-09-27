import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import nltk

# Download stopwords if needed
nltk.download('stopwords')
from nltk.corpus import stopwords

# ---------------------------
# Function to extract text from PDF
# ---------------------------
def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="AI CV Screening", page_icon="📄", layout="wide")
st.title("📄 AI-Powered CV Screening")

st.markdown("Upload CVs (PDF/TXT) and paste a job description. The system will rank candidates by best match.")

# Upload CVs
uploaded_files = st.file_uploader("Upload CVs", type=["pdf", "txt"], accept_multiple_files=True)

# Job description input
job_description = st.text_area("Paste Job Description", height=150)

# Define key skills (default, can edit)
skills_input = st.text_input(
    "Enter key skills (comma-separated)", 
    value="sql, python, power bi, tableau, excel, data cleaning"
)
key_skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]


# Skill weight slider
skill_weight = st.slider("Skill Match Weight (%)", 0, 100, 30)
tfidf_weight = 100 - skill_weight

if st.button("Process CVs"):
    if not uploaded_files or not job_description.strip():
        st.warning("Please upload CVs and enter a job description.")
    else:
        cvs = {}
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                text = read_pdf(file)
            else:
                text = file.read().decode("utf-8")
            cvs[file.name] = text

        # Preprocess
        stop_words = stopwords.words('english')
        documents = list(cvs.values()) + [job_description]
        names = list(cvs.keys())

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Similarity scores
        similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

        # Results
        results = []
        for i, name in enumerate(names):
            cv_text = cvs[name].lower()
            matched = [skill for skill in key_skills if skill in cv_text]
            missing = [skill for skill in key_skills if skill not in cv_text]
            
            # Scores
            sim_score = similarities[i][0] * 100
            skill_match_pct = (len(matched) / len(key_skills)) * 100 if key_skills else 0
            
            # Weighted final score
            final_score = round(
                (tfidf_weight/100) * sim_score + (skill_weight/100) * skill_match_pct, 2
            )
            
            results.append((name, f"{final_score}%", f"{skill_match_pct:.1f}%", ", ".join(matched) if matched else "None", ", ".join(missing) if missing else "None"))

        df = pd.DataFrame(results, columns=["Candidate", "Final Score", "Skill Match %", "Matched Skills", "Missing Skills"])
        df = df.sort_values(by="Final Score", ascending=False)

        st.success("✅ CV Screening Completed!")
        st.dataframe(df, use_container_width=True)

        # Highlight best candidate
        best_candidate = df.iloc[0]
        st.markdown(f"### 🏆 Best Match: **{best_candidate['Candidate']}** ({best_candidate['Final Score']})")

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="cv_screening_results.csv",
            mime="text/csv",
        )
