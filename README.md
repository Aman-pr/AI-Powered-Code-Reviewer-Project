# AI-Powered Resume Analysis System

A full-featured system to analyze resumes against job descriptions using AI embeddings and LLMs. Provides detailed, actionable feedback for hiring decisions.

---

## Features

- **Single Resume Analysis:** Evaluate one resume against a job description.  
- **Batch Processing:** Analyze multiple resumes in a folder against a single job description.  
- **PDF Text Extraction:** Extract resume content using `pdfplumber`.  
- **AI Semantic Similarity:** Compare resumes with job descriptions using Hugging Face embeddings.  
- **Detailed AI Reviews:** Generate structured feedback using Groq LLM.  
- **Professional Output:** JSON export with similarity scores, interpretations, and detailed review.  
- **Configurable Settings:** Centralized configuration for API keys, models, thresholds, and output directories.  

---

## Repository Structure
your-repo-name/
├── resume_analyzer.py # Main application
├── config.py # Configuration management
├── requirements.txt # Python dependencies
├── README.md # This documentation
├── sample_data/ # Sample resume & job description files
└── results/ # Output results directory


## Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/cloudburst-prediction.git
   cd cloudburst-prediction

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Set up API keys::
   ```bash
   export HF_API_KEY="your_huggingface_api_key"
   export GROQ_API_KEY="your_groq_api_key"
---

## Usage

### Single Resume Analysis
    python resume_analyzer.py analyze --resume sample_data/sample_resume.pdf --job sample_data/job_description.txt --output results/analysis.json


### Batch Resume Analysis
    python resume_analyzer.py batch --resume-dir ./sample_data/resumes --job sample_data/job_description.txt --output-dir results


---

## Configuration

The system uses `config.py` for centralized settings:
- Embedding model  
- Groq LLM model  
- Temperature & max tokens  
- Output directories  
- Score thresholds & interpretations  

---

## Output

The results include:
- Resume & Job Description previews  
- Similarity score (0.0 - 1.0)  
- Score interpretation  
- Detailed AI review with:
  - Overall assessment  
  - Strengths  
  - Gaps & weaknesses  
  - Recommendations  
  - Hiring verdict  

---

## Dependencies

- Python >= 3.10  
- pdfplumber  
- huggingface-hub  
- groq  
- numpy  

### Install via:
    pip install -r requirements.txt

---

## Author

Aman-pr – AI & Engineering Enthusiast  

---

## License

This project is licensed under the MIT License. See LICENSE for details.
