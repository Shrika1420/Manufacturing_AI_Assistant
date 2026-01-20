# Manufacturing Knowledge Assistant 🏭

An AI-powered employee support system designed for **semiconductor manufacturing environments**, providing instant, accurate answers to HR, safety, onboarding, and equipment-related questions.

This assistant reduces time spent searching manuals, improves onboarding efficiency, and supports safe, compliant operations on the factory floor.

---

## Overview

Manufacturing environments rely heavily on dense documentation across HR policies, safety protocols, and equipment manuals. New hires and operators often struggle to quickly find the right information, leading to delays, errors, and reduced productivity.

The **Manufacturing Knowledge Assistant** solves this by acting as a centralized, conversational knowledge base that delivers precise answers in seconds — 24/7.

---

## Key Features

- **AI-powered question answering**
  - HR policies (vacation, benefits, time-off procedures)
  - Safety protocols (PPE, cleanroom rules, chemical handling)
  - Equipment operations (error codes, troubleshooting, maintenance)
  - Onboarding guidance for new employees

- **Knowledge base ingestion**
  - Processes multiple document sections from HR, Safety, and Operations manuals
  - Ensures answers are grounded in internal documentation

- **Interactive dashboard**
  - Live statistics (questions answered, uptime)
  - Performance metrics (response time, accuracy feedback)
  - Recent and suggested questions for faster discovery

- **User feedback loop**
  - “Helpful / Not Helpful” ratings to evaluate answer quality
  - Performance insights based on real usage

---

## Screenshots

### Main Dashboard & System Overview
Shows the AI assistant interface, knowledge base readiness, and supported domains.

<img width="1919" height="897" alt="Screenshot 2026-01-20 102114" src="https://github.com/user-attachments/assets/7701a996-769e-46bc-9348-6ec32d4f39a6" />


---

### Example Response: First-Day Onboarding
Demonstrates structured, actionable answers sourced from internal onboarding documents.

<img width="1919" height="897" alt="Screenshot 2026-01-20 102234" src="https://github.com/user-attachments/assets/fc43ea8e-3d45-40d5-ae4d-993c8d7864e4" />


---

### Suggested & Recent Questions
Quick-access prompts to guide users toward common HR, safety, and equipment queries.

<img width="423" height="900" alt="Screenshot 2026-01-20 102249" src="https://github.com/user-attachments/assets/bcc36fae-d9c6-4149-9931-675b36f7dd4b" />


---

## Tech Stack

- **Backend:** Python  
- **AI / NLP:** OpenAI API  
- **Environment Management:** `python-dotenv`  
- **Frontend:** Web-based UI (dashboard + chat interface)  
- **Deployment:** Local / cloud-ready architecture  

---

## Environment Variables

This project uses environment variables for sensitive configuration.

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here.

---
#How to run this locally
## Clone the repository
git clone https://github.com/your-username/Manufacturing_AI_Assistant.git
cd Manufacturing_AI_Assistant

## Install dependencies
pip install -r requirements.txt

## Add environment variables
touch .env
## Add OPENAI_API_KEY to .env

## Run the application
python app.py
```

---
# Business Impact
- Faster onboarding for new employees
- Reduced dependency on manual document searches
- Improved safety compliance through instant access to protocols
- Scales across teams without additional HR or support overhead

---
# Future Improvements
- Role-based access (HR vs Operator vs Engineer)
- Document versioning and update tracking
- Analytics dashboard for most-searched topics
- Integration with internal ticketing or HR systems
- Multilingual support for global manufacturing teams

---
# 📌 Disclaimer
This project is a demonstration system using sample manufacturing documentation and does not represent any specific company’s proprietary data.


