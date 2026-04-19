# 🛡️ ChurnPredictor AI
### Enterprise-Grade Customer Retention Intelligence Platform
> **End-Semester GenAI & Agentic AI Project**  
> RAG · LangChain · Groq / Llama-3.1 · FAISS · scikit-learn · Streamlit

---

## 📌 Overview

**ChurnPredictor AI** is an end-to-end **Agentic AI** system that predicts telecom customer churn and autonomously generates personalised retention strategies using:

| Layer | Technology |
|---|---|
| **ML Prediction** | Logistic Regression + Decision Tree (scikit-learn) |
| **RAG** | FAISS vector store + `all-MiniLM-L6-v2` sentence embeddings |
| **LLM** | Groq / **Llama-3.1-8b-instant** via LangChain |
| **Agentic Layer** | `RetentionAgent` orchestrates ML → RAG → LLM pipeline |
| **Frontend** | Streamlit with dark theme, Plotly gauges, session state |

---

## 🏗️ Architecture

```
app.py                          ← Streamlit entry point
├── agent/
│   ├── retention_agent.py      ← Agentic AI core (LangChain + Groq)
│   └── prompts.py              ← LangChain PromptTemplate definitions
├── ml_pipeline/
│   ├── preprocessing.py        ← LabelEncoder + StandardScaler pipeline
│   ├── model_trainer.py        ← Train LR + DT, save to .pkl
│   └── prediction.py           ← ModelLoader + ChurnPredictor
├── rag/
│   ├── embeddings.py           ← Sentence Transformer embedding generator
│   ├── vector_store.py         ← FAISS IndexFlatL2 wrapper
│   └── retriever.py            ← k-NN similarity search + KB builder
├── ui/
│   └── components.py           ← All Streamlit UI components
├── utils/
│   ├── config.py               ← Paths, feature lists, model config
│   └── helpers.py              ← Logging, risk levels, formatters
├── data/
│   └── telco_churn.csv         ← IBM Telco Customer Churn dataset
└── models/
    └── churn_model.pkl         ← Trained models + fitted preprocessor
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/GenAi_Project.git
cd GenAi_Project
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` and add your **Groq API key** (free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=gsk_your_key_here
```

> **No key?** The app still runs in rule-based fallback mode. The LLM panel simply won't appear.

---

## 🚀 Running the App

### (Optional) Pre-train models
```bash
python train.py
```
> Trains models and builds FAISS knowledge base. Skippable — `app.py` trains automatically on first run if `models/churn_model.pkl` is missing.

### Launch Streamlit
```bash
streamlit run app.py
```
Open **http://localhost:8501**

---

## 🧠 Agentic Workflow

```
User Input (Customer Profile)
        │
        ▼
ChurnPredictor.predict()           ← scikit-learn ML inference
        │  prediction + probability
        ▼
RetentionAgent.analyze_churn_risk()
        ├── _extract_risk_factors()      ← deterministic feature analysis
        ├── RAGRetriever.retrieve(k=5)   ← FAISS k-NN: similar historical cases
        └── LLM call via LangChain
                provider:  Groq
                model:     llama-3.1-8b-instant
                prompt:    customer_profile + risk_factors + RAG_cases
                output:    root_cause + prioritised_actions + outcome_prediction
        │
        ▼
Structured Results → Streamlit UI
        ├── Risk gauge (Plotly Indicator)
        ├── 🤖 LLM Reasoning panel (expandable)
        ├── Priority-ranked retention recommendations
        └── Similar RAG-retrieved historical cases
```

---

## 📊 Model Performance (80/20 hold-out)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~80% | ~67% | ~55% | ~60% |
| Decision Tree | ~78% | ~60% | ~52% | ~56% |

---

## 🔑 Key Features

| Feature | Details |
|---|---|
| **LLM-powered recommendations** | Groq / Llama-3.1-8b-instant via LangChain |
| **RAG similarity search** | FAISS + `all-MiniLM-L6-v2`, 500-1000 case KB |
| **Agentic reasoning trace** | Full LLM reasoning shown in UI |
| **Graceful degradation** | Rule-based fallback if no API key |
| **Session persistence** | All results survive button clicks (st.session_state) |
| **Batch prediction** | CSV upload → risk distribution + downloadable results |
| **Dark-theme UI** | Plotly gauges, metric cards, expandable panels |

---

## 🌐 Deployment

### Streamlit Community Cloud
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add `GROQ_API_KEY` in Secrets (Settings → Secrets)

### HuggingFace Spaces
1. Create a new Space → **Streamlit** SDK
2. Push code
3. Add `GROQ_API_KEY` in Space Settings → Repository secrets

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit ≥ 1.28 |
| ML | scikit-learn (LR + DT), pandas, numpy |
| LLM | LangChain + langchain-groq + Groq API (Llama-3.1-8b-instant) |
| RAG | sentence-transformers, FAISS (faiss-cpu) |
| Visualisation | Plotly |
| Dataset | IBM Telco Customer Churn (7,043 records, 20 features) |

---

## 👤 Author

**Anuj Upadhyay**  
End-Semester GenAI & Agentic AI Project
