# Architecture / Workflow

```text
Upload PDF/DOCX
      |
      v
Extract text (pypdf / python-docx)
      |
      v
Clean and normalize text
      |
      v
Detect skills from dictionary
      |
      v
Compare with predefined job-role skills
      |
      v
TF-IDF cosine similarity + skill coverage
      |
      v
Rank roles and recommend top 3
      |
      v
Show missing skills + learning roadmap
      |
      v
Download analysis report
```
