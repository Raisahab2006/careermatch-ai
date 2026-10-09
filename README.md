# CareerMatch AI — Resume Analyzer & Job Matching Platform

A beginner-friendly portfolio MVP that compares a resume with a target job description. It extracts text from PDF/DOCX/TXT files, detects skills from a curated taxonomy, estimates skill coverage and TF-IDF text similarity, and provides practical resume recommendations.

## Features

- Upload PDF, DOCX, or TXT resumes
- Paste a job description and optionally name the target role
- Detect skills by category
- Show skills found in both documents and skills not detected in the resume
- Calculate a transparent heuristic match indicator
- Generate recommendations and download a text report
- Streamlit interface with no API key required

## Important limitations

- The match score is an experimental heuristic, **not an actual ATS score or hiring probability**.
- The skill taxonomy is curated and incomplete. It may miss synonyms, context, and skills expressed in different ways.
- TF-IDF similarity measures word overlap patterns, not deep semantic understanding.
- Scanned/image-only PDFs may return little or no text; OCR is not included in this MVP.
- The app does not intentionally save uploaded files to disk, but do not upload sensitive documents to a deployment you do not control.

## Requirements

- Python 3.10 or newer recommended
- pip

## Run locally (macOS / Linux)

```bash
cd careermatch-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Then run `streamlit run app.py`.

## How to use

1. Open the local URL printed by Streamlit (usually `http://localhost:8501`).
2. Upload a resume as PDF, DOCX, or TXT.
3. Enter the target job title if you want it in the report.
4. Paste the full job description.
5. Click **Analyze resume**.
6. Review matched skills, gaps, recommendations, and download the report.

## Suggested GitHub repository description

`AI-powered resume and job-description matching dashboard built with Python, Streamlit, scikit-learn, and document parsing.`

## Suggested next improvements

1. Add unit tests for document extraction, skill detection, and score calculation.
2. Expand the skill taxonomy and support configurable aliases.
3. Add a benchmark set of resume/job-description pairs and evaluate results.
4. Compare TF-IDF against sentence embeddings and explain the trade-offs.
5. Add a secure database-backed user history only if required, with explicit retention controls.
6. Add optional LLM-powered feedback with citations to the exact resume sections; keep API keys server-side.
7. Add automated tests and Docker deployment.

## Suggested CV bullet (use only after you have run and verified the app)

> Built CareerMatch AI, a resume-to-job-description analysis dashboard using Python, Streamlit, and scikit-learn. Implemented PDF/DOCX text extraction, categorized skill detection, TF-IDF similarity, skill-gap reporting, and downloadable recommendations.
