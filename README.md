
# 🌍 Geo-Compliance AI Assistant

An AI-powered compliance assistant that leverages **Retrieval-Augmented Generation (RAG)** to automatically assess whether product features require geo-specific compliance logic, provide clear reasoning, generate actionable recommendations, and maintain audit-ready evidence trails.

---

## ⚙️ Setup Instructions

### macOS
```bash
# 1. Set up Python environment (using venv)
python3 -m venv venv

# 2. Activate environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py
```

### Windows (PowerShell)
```powershell
# 1. Set up Python environment (using venv)
python -m venv venv

# 2. Activate environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py
```

---

## 1. Inspiration

### 1.1 Problem Statement
As corporations or organizations expand their business to global scope, services they offer must comply with a complex and constantly evolving set of geo-specific regulations. Manually identifying and managing these regulatory requirements is time-consuming, error-prone, and costly. Failure to meet these compliance obligations can result in significant legal and financial consequences, exposing organizations to substantial risks.

### 1.2 Motivations and Aim
With the rise of generative AI and retrieval-based systems, we saw an opportunity to build a trusted AI assistant for legal insights. This is not to replace lawyers, but to augment them, reduce research friction, and empower individuals to make more informed decisions, from a higher level of surveillance.

We aim to build a working system that:
- ✅ Flag features requiring geo-specific compliance logic and provide clear reasoning  
- 📑 Keep evidence trails to streamline regulatory inquiries and audits  
- ⚖️ Provide actionable recommendations for compliance improvement or violation handling  
- 🔄 Learn from human feedback to continuously enhance accuracy and precision  

---

## 2. Features

### 2.1 Overview
Our system is built around a **Retrieval-Augmented Generation (RAG)** model designed to assist with geo-specific compliance analysis.

The architecture combines **knowledge retrieval** with **generative reasoning**:
- **Data Preparation**: Curated laws, guidelines, and terminologies processed into embeddings.  
- **Query & Context**: User queries are embedded and searched in vector DB to retrieve top-k relevant documents.  
- **Generative Model**: Retrieved documents + query are passed into an LLM for reasoning.  
- **Structured Outputs**: Compliance flag, reasoning, recommendations, and evidence are provided.  

### 2.2 Cached Legal Data Retrieval
- 🗂️ Fetch once, reuse later (cache legal docs in JSON).  
- 🔄 Refresh every 30 days to balance updates vs performance.  
- ⚡ Fail-safe re-fetch if cache invalid.  
- 📝 Metadata tracked (source, timestamp, law type).  

### 2.3 Context-Aware Smart Chunking
- ✂️ Customized chunking using **law-specific separators** (e.g., Articles, Sections).  
- 📏 Chunk size = 1500 chars, overlap = 200 chars.  
- 🧩 Metadata enriched with law type, source URL, last updated.  
- 💾 Stored in vector DB for retrieval.  

### 2.4 Query Translation & RAG Fusion
- 🔍 **Decomposition**: Break queries into sub-queries.  
- 🌐 **Generalization**: Create broader queries.  
- ⚡ **Enhancement**: Add compliance-specific keywords.  
- 📊 **Fusion**: Apply Reciprocal Rank Fusion (RRF) to merge results, remove duplicates, and feed top-8 chunks to LLM.  

### 2.5 Automated Feedback Integration
- 🔄 **Closed-loop improvement**: Human feedback integrated into future analyses.  
- ✅ **Feedback validation**: LLM judge filters incoherent/irrelevant inputs.  
- 💡 **Vector storage**: Feedback embedded and stored in Chroma DB.  
- 🧭 **Dual-mode workflow**:  
  - Mode 1: Feedback Mode → Single-feature correction.  
  - Mode 2: Batch Mode → Bulk analysis with optional later review.  
