# CV-Evaluator
Overview
An AI-powered system that evaluates image-based resumes of IT professionals and recommends suitable job roles. It leverages OCR and LLMs to assess resume quality and provide career development insights.
Features
•	Extracts text from resume images using Tesseract OCR
•	Analyzes resume content using Ollama's Gemma3:4B LLM
•	Matches candidate skills to predefined IT job roles
•	Provides a suitability score, strengths, weaknesses, and career advice
System Architecture
1. OCR Text Extraction
Uses `pytesseract` and `Pillow` to extract text from image files.
2. AI Analysis Engine
Utilizes Ollama's `gemma3:4b` model to evaluate resumes on five key dimensions.
3. Job Matching Algorithm
Matches extracted skills with 8 predefined IT job roles using `scikit-learn` and `numpy`.
Supported Formats
• Image formats: PNG, JPG, JPEG, BMP, GIF
• PDF support planned
Job Roles Covered
Job Title	Level	Key Skills
Software Developer	Mid	Python, JavaScript, React, SQL, Git
Data Scientist	Mid	Python, ML, Pandas, NumPy, Statistics
Frontend Developer	Entry	JavaScript, React, HTML, CSS, TypeScript
DevOps Engineer	Senior	Docker, Kubernetes, AWS, Linux, Python
Project Manager	Senior	PM, Agile, Scrum, Leadership
Junior Web Developer	Entry	HTML, CSS, JavaScript, PHP, MySQL
Machine Learning Eng.	Senior	Python, TensorFlow, PyTorch, ML, DL
Business Analyst	Mid	Excel, SQL, Data Analysis, BI, PowerBI
Evaluation Criteria (0–10 scale)
• Technical Skills Match: Skill relevance to job
• Experience Level Fit: Experience suitability
• Industry Knowledge: Domain familiarity
• Education Alignment: Academic background
• Overall Potential: Career trajectory
Suitability Score Interpretation
Score	Interpretation	System Response
9–10	Exceptional Fit	Highly recommended
7–8	Strong Fit	Minor enhancements needed
5–6	Moderate Fit	Suitable with improvement
3–4	Weak Fit	Major development needed
1–2	Poor Fit	Career pivot advised
Output Report
• CV Analysis Summary: Key background and qualifications
• Recommended Job Position: Best-fit title and context
• Scoring Breakdown: Scores and explanations per category
• Career Development Advice: Actionable steps and suggestions
• Justification: Explanation of job match and skills alignment
Technical Requirements
•	Python 3.x
•	`pytesseract`, `Pillow`, `requests`, `scikit-learn`, `numpy`
•	Tesseract OCR installed
•	Ollama server running with `gemma3:4b`
Limitations
•	English-language resumes only
•	IT-related positions only
•	Standard layout formats recommended
•	Creative portfolios and non-technical roles not supported
