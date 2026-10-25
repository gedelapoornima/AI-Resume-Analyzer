import re
from io import BytesIO
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

SKILLS = [
    "python", "java", "c++", "c#", ".net", "sql", "excel", "power bi", "tableau",
    "pandas", "numpy", "matplotlib", "machine learning", "deep learning", "scikit-learn",
    "tensorflow", "pytorch", "nlp", "natural language processing", "transformers",
    "hugging face", "llm", "rag", "opencv", "cnn", "yolo", "fastapi", "flask",
    "streamlit", "docker", "git", "github", "aws", "azure", "linux", "html", "css",
    "javascript", "react", "mongodb", "mysql", "postgresql", "statistics", "data analysis",
    "data visualization", "api", "communication", "problem solving"
]

ROLES = {
    "Data Analyst": ["python", "sql", "excel", "pandas", "power bi", "tableau", "numpy", "statistics", "data analysis", "data visualization"],
    "Machine Learning Engineer": ["python", "machine learning", "scikit-learn", "pandas", "numpy", "fastapi", "docker", "statistics", "git", "sql"],
    "AI Engineer": ["python", "deep learning", "llm", "rag", "api", "fastapi", "docker", "pytorch", "tensorflow", "git"],
    "NLP Engineer": ["python", "nlp", "natural language processing", "transformers", "hugging face", "llm", "pytorch", "tensorflow", "api"],
    "Computer Vision Engineer": ["python", "opencv", "cnn", "yolo", "deep learning", "pytorch", "tensorflow", "numpy", "git"],
    "Python Developer": ["python", "sql", "git", "github", "api", "fastapi", "flask", "docker", "linux"],
}

ROADMAP = {
    "python": "Practice Python fundamentals, functions, and file handling.",
    "sql": "Learn SELECT, JOIN, GROUP BY, subqueries, and database design.",
    "excel": "Practice formulas, lookup functions, pivot tables, and charts.",
    "power bi": "Learn data cleaning, dashboards, and basic DAX.",
    "machine learning": "Study supervised learning, model evaluation, and feature engineering.",
    "scikit-learn": "Build classification and regression projects with scikit-learn.",
    "fastapi": "Create a REST API and test endpoints with sample data.",
    "docker": "Containerize a small Python application.",
    "llm": "Learn LLM basics, prompting, and evaluation.",
    "rag": "Build a small retrieval-augmented generation project.",
    "nlp": "Practice tokenization, text cleaning, and text classification.",
    "transformers": "Study attention and fine-tune a small transformer model.",
    "hugging face": "Explore the Transformers library and model hub.",
    "opencv": "Practice image reading, filtering, and object detection.",
    "cnn": "Learn convolution, pooling, and image classification.",
    "yolo": "Run a small object-detection project using YOLO.",
    "docker": "Learn images, containers, ports, and Dockerfiles.",
    "git": "Practice commits, branches, merges, and GitHub workflows.",
    "aws": "Learn cloud storage, compute basics, and deployment.",
    "statistics": "Review distributions, probability, correlation, and hypothesis testing.",
    "data analysis": "Practice asking questions, cleaning data, and explaining findings.",
    "data visualization": "Create clear charts and explain the insights they show.",
    "api": "Learn HTTP methods, JSON, status codes, and API requests.",
    "pandas": "Practice loading, cleaning, filtering, grouping, and merging data.",
    "numpy": "Practice arrays, vectorized operations, and numerical calculations.",
    "pytorch": "Build and train a small neural network.",
    "tensorflow": "Build a basic neural network and evaluate its results.",
    "tableau": "Build a dashboard with filters and calculated fields.",
    "mysql": "Practice relational tables, joins, indexes, and queries.",
    "mongodb": "Learn documents, collections, CRUD, and aggregation."
}

def extract_text(uploaded_file):
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Please upload a PDF or DOCX file.")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def find_skills(text):
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(skill)
    return sorted(set(found))

def score_roles(resume_text, resume_skills):
    role_names = list(ROLES.keys())
    role_docs = [" ".join(ROLES[role]) for role in role_names]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([resume_text] + role_docs)
    text_scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    rows = []
    for i, role in enumerate(role_names):
        required = ROLES[role]
        found = [s for s in required if s in resume_skills]
        missing = [s for s in required if s not in resume_skills]
        skill_score = (len(found) / len(required)) * 100 if required else 0
        similarity_score = float(text_scores[i]) * 100
        score = round(0.8 * skill_score + 0.2 * similarity_score)
        rows.append({
            "Job Role": role, "Match Score": score,
            "Matched Skills": ", ".join(found) if found else "None detected",
            "Missing Skills": missing, "Required Count": len(required)
        })
    return sorted(rows, key=lambda x: x["Match Score"], reverse=True)

st.title("📄 AI Resume Analyzer & Job Recommendation System")
st.caption("An educational tool that estimates job-role fit from resume skills. Scores are guidance, not hiring decisions.")

with st.sidebar:
    st.header("How to use")
    st.write("1. Upload your PDF/DOCX resume.")
    st.write("2. Review extracted skills.")
    st.write("3. Select a target role.")
    st.write("4. Follow the suggested roadmap.")
    st.info("Your resume is processed in this session and is not intentionally saved by this app.")

uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx"])
if uploaded:
    try:
        raw_text = extract_text(uploaded)
        cleaned = clean_text(raw_text)
        if not cleaned:
            st.error("No readable text found. If this is a scanned PDF, use a text-based PDF or a DOCX file.")
            st.stop()
        skills = find_skills(cleaned)
        results = score_roles(cleaned, skills)
        st.success(f"Successfully read: {uploaded.name}")
        st.subheader("Resume overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Words extracted", len(cleaned.split()))
        c2.metric("Skills found", len(skills))
        c3.metric("Roles compared", len(ROLES))
        st.subheader("Skills detected")
        if skills:
            st.write(", ".join(s.title() for s in skills))
        else:
            st.warning("No skills from the built-in dictionary were detected. Try a resume with a Skills section.")
        st.subheader("Top 3 recommended roles")
        top3 = results[:3]
        cols = st.columns(3)
        for col, item in zip(cols, top3):
            with col:
                st.metric(item["Job Role"], f'{item["Match Score"]}%')
                st.progress(min(item["Match Score"], 100) / 100)
        st.subheader("Role comparison")
        chart_df = pd.DataFrame([{"Job Role": r["Job Role"], "Match Score (%)": r["Match Score"]} for r in results])
        st.bar_chart(chart_df.set_index("Job Role"))
        st.dataframe(chart_df, use_container_width=True, hide_index=True)
        role = st.selectbox("Choose a target role for skill-gap analysis", list(ROLES.keys()))
        selected = next(r for r in results if r["Job Role"] == role)
        st.subheader(f"Skill-gap analysis: {role}")
        st.metric("Estimated match score", f'{selected["Match Score"]}%')
        found = [s for s in ROLES[role] if s in skills]
        missing = [s for s in ROLES[role] if s not in skills]
        left, right = st.columns(2)
        with left:
            st.markdown("**Skills found**")
            st.write(", ".join(s.title() for s in found) if found else "No required skills detected yet.")
        with right:
            st.markdown("**Skills to build**")
            st.write(", ".join(s.title() for s in missing) if missing else "All listed role skills were detected.")
        st.subheader("Suggested learning roadmap")
        if missing:
            for week, skill in enumerate(missing[:4], start=1):
                suggestion = ROADMAP.get(skill, f"Study {skill.title()} and complete a small practical project.")
                st.write(f"**Week {week}: {skill.title()}** — {suggestion}")
        else:
            st.write("Revise the role skills and build a project that demonstrates them.")
        report = [
            "AI RESUME ANALYZER - ANALYSIS REPORT",
            f"Resume: {uploaded.name}",
            f"Words extracted: {len(cleaned.split())}",
            f"Skills found: {', '.join(skills) if skills else 'None detected'}",
            "",
            "ROLE MATCHES:"
        ]
        for r in results:
            report.append(f'{r["Job Role"]}: {r["Match Score"]}%')
        report += ["", f"TARGET ROLE: {role}", f'Match score: {selected["Match Score"]}%',
                   f'Matched skills: {", ".join(found) if found else "None"}',
                   f'Missing skills: {", ".join(missing) if missing else "None"}',
                   "", "Note: Scores are estimates for learning guidance, not recruiter decisions."]
        st.download_button("Download analysis report (.txt)", "\n".join(report), file_name="resume_analysis_report.txt", mime="text/plain")
        with st.expander("View extracted resume text"):
            st.text_area("Extracted text", raw_text, height=250)
    except Exception as e:
        st.error(f"Could not process this file: {e}")
else:
    st.info("Upload a PDF or DOCX resume to begin.")
    st.markdown("**Included roles:** Data Analyst, Machine Learning Engineer, AI Engineer, NLP Engineer, Computer Vision Engineer, and Python Developer.")
