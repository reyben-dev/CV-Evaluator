import base64
import requests
import json
import sys
import os
from PIL import Image
import io
import pytesseract
from datetime import datetime
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Tesseract path (adjust based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ollama Configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
VISION_MODEL = "gemma3:4b"  # Using Gemma 3 4B model

# CV Analysis and Job Suggestion Prompt
CV_ANALYSIS_PROMPT = """You are an expert career counselor and job placement specialist. I will provide you with text extracted from a CV/resume, and you need to analyze it and suggest the most suitable job role.

Based on the CV content, provide:

1. **CV Analysis Summary**
2. **Best Suited Job Role** - Suggest ONE specific job title that matches their skills and experience
3. **Suitability Score** - Rate from 1-10 why they're suitable for this role
4. **Detailed Scoring Criteria** (each out of 10):
   - Technical Skills Match (0-10)
   - Experience Level Fit (0-10) 
   - Industry Knowledge (0-10)
   - Education Alignment (0-10)
   - Overall Potential (0-10)

**Required Output Format:**
CV_SUMMARY: [Brief analysis of candidate's background]

RECOMMENDED_JOB: [Specific job title]

OVERALL_SUITABILITY_SCORE: [X/10]

DETAILED_SCORING:
- Technical Skills Match: [X/10] - [Explanation]
- Experience Level Fit: [X/10] - [Explanation]
- Industry Knowledge: [X/10] - [Explanation] 
- Education Alignment: [X/10] - [Explanation]
- Overall Potential: [X/10] - [Explanation]

JUSTIFICATION: [Detailed paragraph explaining why this job is perfect for them]

NEXT_STEPS: [3-4 actionable recommendations for career development]

Now analyze this CV/resume text:"""

JOB_DATABASE = [
    {
        "id": 1,
        "title": "Software Developer",
        "company": "TechCorp Inc.",
        "required_skills": ["python", "javascript", "react", "sql", "git"],
        "experience_level": "Mid",
        "industry": "Technology",
        "description": "Develop web applications using modern frameworks",
        "education_requirement": "Bachelor"
    },
    {
        "id": 2,
        "title": "Data Scientist",
        "company": "DataFlow Solutions",
        "required_skills": ["python", "machine learning", "pandas", "numpy", "sql", "statistics"],
        "experience_level": "Mid",
        "industry": "Data Analytics",
        "description": "Analyze large datasets and build predictive models",
        "education_requirement": "Master"
    },
    {
        "id": 3,
        "title": "Frontend Developer",
        "company": "WebDesign Pro",
        "required_skills": ["javascript", "react", "html", "css", "typescript"],
        "experience_level": "Entry",
        "industry": "Technology",
        "description": "Create responsive and interactive user interfaces",
        "education_requirement": "Bachelor"
    },
    {
        "id": 4,
        "title": "DevOps Engineer",
        "company": "CloudTech Systems",
        "required_skills": ["docker", "kubernetes", "aws", "linux", "python", "terraform"],
        "experience_level": "Senior",
        "industry": "Technology",
        "description": "Manage cloud infrastructure and deployment pipelines",
        "education_requirement": "Bachelor"
    },
    {
        "id": 5,
        "title": "Project Manager",
        "company": "Business Solutions Ltd",
        "required_skills": ["project management", "agile", "scrum", "leadership", "communication"],
        "experience_level": "Senior",
        "industry": "Business",
        "description": "Lead cross-functional teams and manage project deliverables",
        "education_requirement": "Master"
    },
    {
        "id": 6,
        "title": "Junior Web Developer",
        "company": "StartupHub",
        "required_skills": ["html", "css", "javascript", "php", "mysql"],
        "experience_level": "Entry",
        "industry": "Technology",
        "description": "Build and maintain websites for small businesses",
        "education_requirement": "Bachelor"
    },
    {
        "id": 7,
        "title": "Machine Learning Engineer",
        "company": "AI Innovations",
        "required_skills": ["python", "tensorflow", "pytorch", "machine learning", "deep learning"],
        "experience_level": "Senior",
        "industry": "Artificial Intelligence",
        "description": "Develop and deploy ML models for production systems",
        "education_requirement": "Master"
    },
    {
        "id": 8,
        "title": "Business Analyst",
        "company": "Corporate Consulting",
        "required_skills": ["excel", "sql", "data analysis", "business intelligence", "powerbi"],
        "experience_level": "Mid",
        "industry": "Business",
        "description": "Analyze business processes and provide strategic insights",
        "education_requirement": "Bachelor"
    }
]

class CVJobMatcher:
    def __init__(self, model_name=VISION_MODEL):
        self.model_name = model_name
        self.endpoint = OLLAMA_ENDPOINT
        self.job_database = JOB_DATABASE
    
    def extract_text_from_image(self, image_path):
        """Extract text from CV image using OCR"""
        try:
            image = Image.open(image_path)
            # Enhance image for better OCR
            image = image.convert('RGB')
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
        except Exception as e:
            return f"[OCR Error: {str(e)}]"
    
    def preprocess_cv_text(self, text):
        """Clean and preprocess extracted text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def analyze_cv_with_llm(self, cv_text):
        """Use LLM to analyze CV and extract structured information"""
        try:
            full_prompt = CV_ANALYSIS_PROMPT + "\n\n" + cv_text

            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_ctx": 4096
                }
            }

            print("[INFO] Analyzing CV with AI model...")
            response = requests.post(self.endpoint, json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No analysis response received.")
        
        except requests.RequestException as e:
            return f"[ERROR] Ollama API request failed: {str(e)}"
        except Exception as e:
            return f"[ERROR] CV analysis failed: {str(e)}"
    
    def extract_skills_from_text(self, text):
        """Extract technical and soft skills from CV text using keyword matching"""
        technical_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'sql', 'mysql', 'postgresql', 'mongodb', 'html', 'css', 'php',
            'machine learning', 'ai', 'data science', 'pandas', 'numpy',
            'tensorflow', 'pytorch', 'docker', 'kubernetes', 'aws', 'azure',
            'git', 'linux', 'windows', 'excel', 'powerbi', 'tableau'
        ]
        
        soft_skills = [
            'communication', 'leadership', 'teamwork', 'problem solving',
            'project management', 'agile', 'scrum', 'time management',
            'analytical thinking', 'creativity', 'adaptability'
        ]
        
        text_lower = text.lower()
        found_technical = [skill for skill in technical_skills if skill in text_lower]
        found_soft = [skill for skill in soft_skills if skill in text_lower]
        
        return found_technical, found_soft
    
    def calculate_job_match_score(self, cv_skills, job_skills):
        """Calculate matching score between CV skills and job requirements"""
        if not cv_skills or not job_skills:
            return 0.0
        
        # Convert to sets for easier comparison
        cv_set = set([skill.lower() for skill in cv_skills])
        job_set = set([skill.lower() for skill in job_skills])
        
        # Calculate Jaccard similarity
        intersection = cv_set.intersection(job_set)
        union = cv_set.union(job_set)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def parse_job_suggestion(self, cv_analysis):
        """Parse the AI-generated job suggestion and scoring"""
        try:
            # Extract key information from the AI response
            lines = cv_analysis.split('\n')
            
            job_suggestion = {
                'recommended_job': 'Not specified',
                'overall_score': '0/10',
                'detailed_scores': {},
                'justification': 'No justification provided',
                'next_steps': []
            }
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                if line.startswith('RECOMMENDED_JOB:'):
                    job_suggestion['recommended_job'] = line.replace('RECOMMENDED_JOB:', '').strip()
                
                elif line.startswith('OVERALL_SUITABILITY_SCORE:'):
                    job_suggestion['overall_score'] = line.replace('OVERALL_SUITABILITY_SCORE:', '').strip()
                
                elif 'Technical Skills Match:' in line:
                    job_suggestion['detailed_scores']['technical'] = line.split(':')[1].strip()
                elif 'Experience Level Fit:' in line:
                    job_suggestion['detailed_scores']['experience'] = line.split(':')[1].strip()
                elif 'Industry Knowledge:' in line:
                    job_suggestion['detailed_scores']['industry'] = line.split(':')[1].strip()
                elif 'Education Alignment:' in line:
                    job_suggestion['detailed_scores']['education'] = line.split(':')[1].strip()
                elif 'Overall Potential:' in line:
                    job_suggestion['detailed_scores']['potential'] = line.split(':')[1].strip()
                
                elif line.startswith('JUSTIFICATION:'):
                    # Get the justification paragraph
                    justification_start = i + 1
                    justification_lines = []
                    for j in range(justification_start, len(lines)):
                        if lines[j].strip().startswith('NEXT_STEPS:'):
                            break
                        justification_lines.append(lines[j].strip())
                    job_suggestion['justification'] = ' '.join(justification_lines)
                
                elif line.startswith('NEXT_STEPS:'):
                    # Get next steps
                    steps_start = i + 1
                    for j in range(steps_start, len(lines)):
                        step_line = lines[j].strip()
                        if step_line and (step_line.startswith('-') or step_line.startswith('1.') or step_line.startswith('2.')):
                            job_suggestion['next_steps'].append(step_line)
            
            return job_suggestion
            
        except Exception as e:
            return {
                'recommended_job': 'Analysis Error',
                'overall_score': '0/10',
                'detailed_scores': {},
                'justification': f'Error parsing analysis: {str(e)}',
                'next_steps': []
            }
    
    def generate_recommendation_report(self, cv_analysis, cv_text):
        """Generate a comprehensive job recommendation report"""
        job_suggestion = self.parse_job_suggestion(cv_analysis)
        
        report = []
        report.append("=" * 80)
        report.append("CV ANALYSIS AND JOB RECOMMENDATION REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("RECOMMENDED JOB POSITION:")
        report.append("-" * 40)
        report.append(f"🎯 {job_suggestion['recommended_job']}")
        report.append("")
        
        report.append("OVERALL SUITABILITY SCORE:")
        report.append("-" * 40)
        report.append(f"📊 {job_suggestion['overall_score']}")
        report.append("")
        
        report.append("DETAILED SCORING BREAKDOWN:")
        report.append("-" * 40)
        if job_suggestion['detailed_scores']:
            for category, score in job_suggestion['detailed_scores'].items():
                report.append(f"• {category.title()}: {score}")
        else:
            report.append("• Detailed scores not available")
        report.append("")
        
        report.append("WHY THIS JOB IS PERFECT FOR YOU:")
        report.append("-" * 40)
        report.append(job_suggestion['justification'])
        report.append("")
        
        report.append("CAREER DEVELOPMENT RECOMMENDATIONS:")
        report.append("-" * 40)
        if job_suggestion['next_steps']:
            for step in job_suggestion['next_steps']:
                report.append(step)
        else:
            report.append("• Focus on strengthening core technical skills")
            report.append("• Build a portfolio showcasing your projects")
            report.append("• Network within your target industry")
            report.append("• Consider relevant certifications")
        report.append("")
        
        report.append("FULL AI ANALYSIS:")
        report.append("-" * 40)
        report.append(cv_analysis)
        report.append("")
        
        return "\n".join(report)
    
    def process_cv(self, image_path):
        """Main function to process CV and generate job recommendations"""
        try:
            print("[INFO] Extracting text from CV image...")
            cv_text = self.extract_text_from_image(image_path)
            
            if cv_text.startswith("[OCR Error"):
                return cv_text
            
            cv_text = self.preprocess_cv_text(cv_text)
            print(f"[INFO] Extracted {len(cv_text)} characters of text")
            
            print("[INFO] Analyzing CV and generating job suggestion...")
            cv_analysis = self.analyze_cv_with_llm(cv_text)
            
            if cv_analysis.startswith("[ERROR]"):
                return cv_analysis
            
            print("[INFO] Creating recommendation report...")
            report = self.generate_recommendation_report(cv_analysis, cv_text)
            
            return report
            
        except Exception as e:
            return f"[ERROR] CV processing failed: {str(e)}"
    
    def save_report(self, report, image_path, output_dir="cv_reports"):
        """Save the analysis report to a file"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            report_filename = f"{base_name}_job_recommendations.txt"
            report_path = os.path.join(output_dir, report_filename)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"[INFO] Report saved to: {report_path}")
            return report_path
        
        except Exception as e:
            print(f"[WARNING] Failed to save report: {str(e)}")
            return None

def validate_image_file(image_path):
    """Validate if the provided file is a valid image"""
    if not os.path.exists(image_path):
        return False, "File does not exist"
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, "Valid image file"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def main():
    if len(sys.argv) != 2:
        print("=" * 60)
        print("CV/RESUME ANALYSIS AND JOB MATCHING SYSTEM")
        print("=" * 60)
        print("Usage: python cv_job_matcher.py <cv_image_path>")
        print("\nExample:")
        print("  python cv_job_matcher.py my_resume.png")
        print("\nSupported formats: PNG, JPG, JPEG, BMP, GIF, PDF")
        print("\nPrerequisites:")
        print("  1. Install Tesseract OCR")
        print("  2. Install required Python packages:")
        print("     pip install pillow pytesseract requests scikit-learn")
        print("  3. Make sure Ollama is running with Gemma3 model:")
        print("     ollama pull gemma3:4b")
        print("     ollama serve")
        print("=" * 60)
        sys.exit(1)

    cv_image_path = sys.argv[1]
    is_valid, message = validate_image_file(cv_image_path)
    if not is_valid:
        print(f"[ERROR] {message}")
        sys.exit(1)

    print("=" * 60)
    print("CV/RESUME ANALYSIS AND JOB MATCHING SYSTEM")
    print("=" * 60)
    print(f"Processing: {os.path.basename(cv_image_path)}")
    print("=" * 60)

    matcher = CVJobMatcher()
    result = matcher.process_cv(cv_image_path)

    if result.startswith("[ERROR]") or result.startswith("[OCR Error"):
        print(result)
        sys.exit(1)

    print("\n" + result)
    matcher.save_report(result, cv_image_path)

if __name__ == "__main__":
    main()