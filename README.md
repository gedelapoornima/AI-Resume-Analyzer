# AI Resume Analyzer and Job Recommendation System

## Overview
A Streamlit educational application that reads PDF/DOCX resumes, detects job-related skills, compares the resume with six predefined job roles, ranks roles, identifies missing skills, and suggests a simple learning roadmap.

## Features
- PDF and DOCX resume upload
- Text extraction and cleaning
- Keyword-based skill detection
- TF-IDF and cosine similarity combined with skill coverage for estimated role-match scores
- Top three role recommendations
- Skill-gap analysis and learning roadmap
- Downloadable text report

## Setup (Windows)
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
If `py` is unavailable, use `python -m venv .venv` if Python is on PATH. Alternatively use your installed Python interpreter to create the environment.

## Use
1. Open the local Streamlit URL shown in the terminal.
2. Upload a text-based PDF or DOCX resume.
3. Review detected skills and the top three recommended roles.
4. Select a role to view missing skills and the learning roadmap.
5. Download the analysis report.

## Limitations and responsible use
This is a learning prototype, not a hiring or rejection tool. Scores are estimates and depend on the built-in skill dictionary and role requirements. Missing keywords do not necessarily mean a person lacks a skill. Do not use gender, age, religion, nationality, photographs, marital status, disability, or other protected personal attributes in scoring. Uploaded resumes are processed in the app session and are not intentionally saved by this application.

## Tech stack
Python, Streamlit, pypdf, python-docx, Pandas, scikit-learn.
