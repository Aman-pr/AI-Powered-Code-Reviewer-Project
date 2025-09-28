#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
import pdfplumber
from datetime import datetime
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq


class ResumeJobAnalyzer:

    def __init__(self, hf_api_key, groq_api_key):
        self.hf_client = InferenceClient(token=hf_api_key)
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.groq_model = "llama-3.3-70b-versatile"

    def extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")

            return text.strip()

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def load_job_description(self, job_path):
        if not os.path.exists(job_path):
            raise FileNotFoundError(f"Job description file not found: {job_path}")

        try:
            with open(job_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                raise ValueError("Job description file is empty")

            return content

        except Exception as e:
            raise Exception(f"Error loading job description: {str(e)}")

    def generate_embeddings(self, texts):
        try:
            embeddings = self.hf_client.feature_extraction(
                texts,
                model=self.embedding_model
            )
            return np.array(embeddings, dtype="float32")

        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    def calculate_similarity(self, resume_text, job_desc):
        try:
            embeddings = self.generate_embeddings([resume_text, job_desc])
            v1, v2 = embeddings[0], embeddings[1]
            similarity = dot(v1, v2) / (norm(v1) * norm(v2))
            return float(similarity)

        except Exception as e:
            raise Exception(f"Error calculating similarity: {str(e)}")

    def generate_detailed_review(self, resume_text, job_desc, similarity_score):
        prompt = f"""You are an expert HR consultant and career coach. Analyze the following resume against the job description and provide a comprehensive evaluation.

JOB DESCRIPTION:
{job_desc}

RESUME:
{resume_text}

SIMILARITY SCORE: {similarity_score:.3f}

Please provide a detailed analysis with the following sections:

1. OVERALL ASSESSMENT:
   - Brief summary of candidate fit
   - Key compatibility score interpretation

2. STRENGTHS:
   - Relevant skills and experience that match the job requirements
   - Notable achievements and qualifications
   - Technical skills alignment

3. GAPS AND WEAKNESSES:
   - Missing skills or experience
   - Areas where the resume doesn't align with job requirements
   - Potential concerns for the hiring manager

4. RECOMMENDATIONS FOR IMPROVEMENT:
   - Specific suggestions to better align with this role
   - Skills to develop or highlight
   - Resume formatting or content suggestions

5. HIRING VERDICT:
   - Recommendation (Strong Match/Good Match/Partial Match/Poor Match)
   - Likelihood of getting an interview
   - Key factors that would influence the hiring decision

Please be constructive, specific, and actionable in your feedback."""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Error generating review: {str(e)}")

    def analyze_resume(self, resume_path, job_desc_path, output_file=None):
        print("🔍 Starting Resume Analysis...")
        print("=" * 50)

        try:
            print("📄 Extracting text from resume PDF...")
            resume_text = self.extract_text_from_pdf(resume_path)
            print(f"   ✓ Extracted {len(resume_text)} characters")

            print("📋 Loading job description...")
            job_desc = self.load_job_description(job_desc_path)
            print(f"   ✓ Loaded {len(job_desc)} characters")

            print("🤖 Computing semantic similarity...")
            similarity_score = self.calculate_similarity(resume_text, job_desc)
            print(f"   ✓ Similarity Score: {similarity_score:.3f}")

            print("💬 Generating AI-powered review...")
            detailed_review = self.generate_detailed_review(resume_text, job_desc, similarity_score)
            print("   ✓ Review generated successfully")

            results = {
                "analysis_date": datetime.now().isoformat(),
                "resume_file": os.path.basename(resume_path),
                "job_description_file": os.path.basename(job_desc_path),
                "similarity_score": similarity_score,
                "score_interpretation": self._interpret_score(similarity_score),
                "detailed_review": detailed_review,
                "resume_preview": resume_text[:300] + "..." if len(resume_text) > 300 else resume_text,
                "job_desc_preview": job_desc[:300] + "..." if len(job_desc) > 300 else job_desc
            }

            self._display_results(results)

            if output_file:
                self._save_results(results, output_file)
                print(f"💾 Results saved to: {output_file}")

            return results

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            return None

    def _interpret_score(self, score):
        if score >= 0.8:
            return "Excellent Match"
        elif score >= 0.7:
            return "Very Good Match"
        elif score >= 0.6:
            return "Good Match"
        elif score >= 0.5:
            return "Fair Match"
        elif score >= 0.4:
            return "Poor Match"
        else:
            return "Very Poor Match"

    def _display_results(self, results):
        print("\n" + "=" * 60)
        print("📊 RESUME ANALYSIS RESULTS")
        print("=" * 60)
        print(f"📅 Analysis Date: {results['analysis_date'][:19]}")
        print(f"📄 Resume File: {results['resume_file']}")
        print(f"📋 Job Description: {results['job_description_file']}")
        print(f"🎯 Similarity Score: {results['similarity_score']:.3f}")
        print(f"📈 Interpretation: {results['score_interpretation']}")
        print(f"\n{'🤖 AI REVIEW':^60}")
        print("-" * 60)
        print(results['detailed_review'])
        print("-" * 60)

    def _save_results(self, results, output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def batch_analyze(self, resume_dir, job_desc_file, output_dir="results"):
        if not os.path.exists(resume_dir):
            print(f"Resume directory not found: {resume_dir}")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pdf_files = [f for f in os.listdir(resume_dir) if f.endswith('.pdf')]

        if not pdf_files:
            print("No PDF files found in resume directory")
            return

        print(f"🔄 Starting batch analysis of {len(pdf_files)} resumes...")

        results_summary = []

        for pdf_file in pdf_files:
            print(f"\n📄 Processing: {pdf_file}")
            resume_path = os.path.join(resume_dir, pdf_file)
            output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_analysis.json")

            result = self.analyze_resume(resume_path, job_desc_file, output_file)

            if result:
                results_summary.append({
                    "file": pdf_file,
                    "score": result["similarity_score"],
                    "interpretation": result["score_interpretation"]
                })

        summary_file = os.path.join(output_dir, "batch_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Batch analysis complete. Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Resume-Job Description Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resume_analyzer.py analyze --resume resume.pdf --job job_desc.txt
  python resume_analyzer.py analyze --resume resume.pdf --job job_desc.txt --output results.json
  python resume_analyzer.py batch --resume-dir ./resumes --job job_desc.txt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze single resume")
    analyze_parser.add_argument("--resume", required=True, help="Path to resume PDF file")
    analyze_parser.add_argument("--job", required=True, help="Path to job description text file")
    analyze_parser.add_argument("--output", help="Output JSON file path (optional)")

    batch_parser = subparsers.add_parser("batch", help="Analyze multiple resumes")
    batch_parser.add_argument("--resume-dir", required=True, help="Directory containing resume PDF files")
    batch_parser.add_argument("--job", required=True, help="Path to job description text file")
    batch_parser.add_argument("--output-dir", default="results", help="Output directory for results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    hf_api_key = os.getenv("HF_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not hf_api_key:
        hf_api_key = input("Enter your Hugging Face API Key: ").strip()

    if not groq_api_key:
        groq_api_key = input("Enter your Groq API Key: ").strip()

    if not hf_api_key or not groq_api_key:
        print(
            "❌ API keys are required. Please set HF_API_KEY and GROQ_API_KEY environment variables or enter them when prompted.")
        return

    try:
        analyzer = ResumeJobAnalyzer(hf_api_key, groq_api_key)

        if args.command == "analyze":
            analyzer.analyze_resume(args.resume, args.job, args.output)

        elif args.command == "batch":
            analyzer.batch_analyze(args.resume_dir, args.job, args.output_dir)

    except Exception as e:
        print(f"❌ Error initializing analyzer: {str(e)}")
        return


if __name__ == "__main__":
    main()
