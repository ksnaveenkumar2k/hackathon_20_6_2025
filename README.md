# hackathon_20_6_2025
## Reality-Matching Simulator

**Overview**

The Reality-Matching Simulator is an AI-powered web application designed to align users' career aspirations with market realities. It analyzes user-provided career goals, professional profiles, and resumes to deliver personalized insights, including gap analysis, micro-OKRs (Objectives and Key Results), and a detailed coaching report. The application leverages advanced AI models, LangChain, and a modern React frontend to provide actionable career guidance.

**Key Features**

Career Aspirations Analysis: Evaluates desired CTC (Cost to Company), target companies, roles, and personal OKRs.

Profile Evaluation: Analyzes user profiles, including LinkedIn, GitHub, LeetCode, and uploaded resumes (PDF/DOCX).

Gap Analysis: Provides a comprehensive breakdown of alignment between aspirations and current market standards.

Actionable Micro-OKRs: Generates a 30-day plan to bridge identified gaps.

AI Coaching Report: Delivers personalized feedback and next steps using AI-driven insights.

Responsive UI: Built with React, Tailwind CSS, and Lucide icons for a modern, user-friendly experience.

**Technologies Used**

**Backend:**

Python 3.8+

LangChain (with Google Generative AI integration)

FAISS for vector search and retrieval-augmented generation (RAG)

SentenceTransformer for embeddings (all-MiniLM-L6-v2)

Django for API framework

LangGraph for workflow orchestration

PyPDFLoader and python-docx for resume parsing

**Frontend:**

React (with TypeScript)

Tailwind CSS for styling

Lucide React for icons

External Dependencies:

Google API Key for Gemini-1.5-flash model

SentenceTransformer model

FAISS vector store

**Environment:**

Node.js (for frontend)

Python environment with pip for backend dependencies

**Prerequisites**

Python 3.8 or higher
Node.js 18 or higher
Google API Key (for Gemini-1.5-flash model)
A PDF or DOCX file (RAG_Dataset.pdf) for market data in the RAG setup
Git for cloning the repository



Here's the structure of your project directory as shown in the image:

```
HACKATHON/
└── hackathon_20_6_20/
    ├── backend/
    │   ├── api/
    │   │   ├── migrations/
    │   │   ├── __init__.py
    │   │   ├── admin.py
    │   │   ├── agents.py
    │   │   ├── apps.py
    │   │   ├── models.py
    │   │   ├── tests.py
    │   │   ├── urls.py
    │   │   └── views.py
    │   ├── backend
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   ├── wsgi.py
    │   ├── utils/
    │   │   └── RAG_Dataset.pdf
    │   ├── .env
    │   ├── db.sqlite3
    │   ├── manage.py
    │   ├── requirements.txt
    └── (other possible directories/files)
```

Would you like help with anything specific related to this structure?


**Installation**

1.Clone the Repository:

```bash

git clone https://github.com/ksnaveenkumar2k/hackathon_20_6_2025.git

Install Python dependencies:

pip install -r requirements.txt

django
langchain
langchain-google-genai
langchain-community
faiss-cpu
sentence-transformers
python-docx
pypdf
pydantic

```
Set up environment variables: Create a .env file in the project root and add:

```bash
GOOGLE_API_KEY=your_google_api_key
RAG_DATASET_PATH=/path/to/RAG_Dataset.pdf
LOG_LEVEL=INFO
``

Run database migrations (if using Django):

python manage.py migrate

Start the Django server:

python manage.py runserver

```


**Frontend Setup:**

```bash

cd frontend
npm install

Start the development server:

npm run dev

```

**Ensure RAG Dataset:**

Place the RAG_Dataset.pdf file in the specified RAG_DATASET_PATH or update the path in the .env file.

**Project Code Architecture Flow:**

[Frontend: Home Component]
       ↓ (POST /api/profile/)
[Backend: Django API (views.py)]
       ↓
[AgentExecutor (agents.py)]
   ├── _setup_rag (RAG with FAISS)
   ├── _build_workflow (StateGraph)
   ├── execute (sync/async)
   │   ├── _aspiration_parser_agent (ReAct)
   │   ├── _parallel_agents
   │   │   ├── _profile_evaluator_agent (resume processing)
   │   │   └── _market_benchmarking_agent (RAG retrieval)
   │   ├── _reality_score_agent
   │   ├── _delta_reducer_agent (micro-OKRs)
   │   └── _self_awareness_coach_agent (report)
   └── (Fallbacks & Monitoring)
       ↓
[JSON Response]
       ↓
[Frontend: RenderEnhancedResult]


