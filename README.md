
# Geo-Compliance AI Assistant

# Setup Instructions
# macOS
# 1. Set up Python environment (using venv)
python3 -m venv venv

# 2. Activate environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py


# Window
# 1. Set up Python environment (using venv)
python -m venv venv

# 2. Activate environment
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py



# 1. Set up Python environment (using venv)
python3 -m venv venv

# 2. Activate environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py


## 1. Inspiration

### 1.1 Problem Statement
As corporations or organizations expand their business to global scope, services they offer must comply with a complex and constantly evolving set of geo-specific regulations. Manually identifying and managing these regulatory requirements is time-consuming, error-prone, and costly. Failure to meet these compliance obligations can result in significant legal and financial consequences, exposing organizations to substantial risks.

### 1.2 Motivations and Aim
With the rise of generative AI and retrieval-based systems, we saw an opportunity to build a trusted AI assistant for legal insights. Our goal aligns with TechJam Track #3, this is not to replace lawyers, but to augment them, reduce research friction, and empower individuals to make more informed decisions, from a higher level of surveillance.

We aim to build a working system that:
- Flag features requiring geo-specific compliance logic and provides clear reasoning for each assessment  
- Keep evidence trails to streamline regulatory inquiries and audits  
- Provide actionable recommendations for compliance improvement or violation handling  
- Learn from human feedback to continuously enhance accuracy and precision  

## 2. Features

### 2.1 Overview
Our system is built around a Retrieval-Augmented Generation (RAG) model designed to assist with geo-specific compliance analysis. The main goal is to automatically determine whether a feature requires location-specific compliance logic, provide clear reasoning, and generate actionable recommendations—all while maintaining audit-ready evidence.

The RAG architecture combines knowledge retrieval with generative reasoning:
- **Data Preparation**: Curated laws, guidelines, and terminologies processed into embeddings.  
- **Query & Context**: User queries are embedded and searched in vector DB to retrieve top-k relevant documents.  
- **Generative Model**: Retrieved documents + query are passed into an LLM for reasoning.  
- **Structured Outputs**: Compliance flag, reasoning, recommendations, and evidence are provided.  

### 2.2 Cached Legal Data Retrieval
- Fetch once, reuse later (cache legal docs in JSON).  
- Refresh every 30 days to balance updates vs performance.  
- Fail-safe re-fetch if cache invalid.  
- Metadata tracked (source, timestamp, law type).  

### 2.3 Context-Aware Smart Chunking
- Customized chunking using **law-specific separators** (e.g., Articles, Sections).  
- Chunk size = 1500 chars, overlap = 200 chars.  
- Metadata enriched with law type, source URL, last updated.  
- Stored in vector DB for retrieval.  

### 2.4 Query Translation & RAG Fusion
- **Decomposition**: Break queries into sub-queries.  
- **Generalization**: Create broader queries.  
- **Enhancement**: Add compliance-specific keywords.  
- **Fusion**: Apply Reciprocal Rank Fusion (RRF) to merge results, remove duplicates, and feed top-8 chunks to LLM.  

### 2.5 Automated Feedback Integration
- **Closed-loop improvement**: Human feedback integrated into future analyses.  
- **Feedback validation**: LLM judge filters incoherent/irrelevant inputs.  
- **Vector storage**: Feedback embedded and stored in Chroma DB.  
- **Dual-mode workflow**:  
  - Mode 1 (Feedback Mode): Single-feature correction.  
  - Mode 2 (Batch Mode): Bulk analysis with optional later review.  

## 3. Challenges

- **Database Storage for Feedback**: Initial JSON/relational storage caused inefficiency, duplication, and scalability issues → switched to vector DB.  
- **Validating Feedback at Scale**: Rule-based validation was brittle → introduced LLM-based Judge module.  
- **Workflow Scalability & UX**: Designed dual-mode feedback vs batch analysis to optimize workload.  

## 4. Accomplishments
- Implemented caching, smart chunking, query fusion, and automated feedback pipeline.  
- Balanced speed, accuracy, and auditability in compliance analysis.  

## 5. What’s Next
- Explore **GraphRAG** for deeper relational reasoning across compliance documents and knowledge bases.  

