Overview
An AI system that reads resume images, evaluates them using an AI model, and recommends the best IT job based on the candidate's skills and experience.
Features
•	Reads resume images using OCR
•	Analyzes resume using AI (Gemma3:4B)
•	Matches skills to job roles
•	Gives feedback and career advice
How It Works
1. Extracts text from image (OCR)
2. Analyzes content using AI model
3. Matches candidate to 1 of 8 IT job roles
Job Roles
• Software Developer (Mid)
• Data Scientist (Mid)
• Frontend Developer (Entry)
• DevOps Engineer (Senior)
• Project Manager (Senior)
• Junior Web Developer (Entry)
• Machine Learning Engineer (Senior)
• Business Analyst (Mid)
Score Criteria (0–10)
• Skills match
• Experience level
• Industry knowledge
• Education fit
• Career potential
Output Includes
• Summary of resume
• Best job match
• Score breakdown
• Advice for improvement
• Why the job fits
Requirements
• Python 3.x
• Tesseract OCR installed
• Ollama with Gemma3:4B running
• Libraries: pytesseract, Pillow, requests, scikit-learn, numpy
Limitations
• English resumes only
• IT jobs only
• Clear resume layouts only
• No creative/non-technical resumes
