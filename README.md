# AI-Powered CV Screening System 📄🤖

This is a Streamlit app that allows HR to upload multiple CVs (PDF/TXT) and a Job Description.  
The system automatically ranks candidates using:
- TF-IDF text similarity
- Skill matching
- Weighted scoring

## Features
✅ Upload multiple CVs  
✅ Paste a Job Description  
✅ Automatic ranking with Final Score  
✅ Matched & Missing Skills  
✅ Download results as CSV  

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run cv_match_app.py
