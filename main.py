# main.py
import os
import json
import bs4
import csv
import time
import re
import hashlib
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.documents import Document
from chromadb import telemetry

# Disable all telemetry
original_capture = telemetry.product_telemetry_client.ProductTelemetryClient.capture

def silent_capture(self, *args, **kwargs):
    """Silently ignore all telemetry events"""
    return None

# Apply the monkey patch
telemetry.product_telemetry_client.ProductTelemetryClient.capture = silent_capture

# Also set environment variable for good measure
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "True"

class ComplianceGuardian:
    def __init__(self, use_cache: bool = True):
        self.legal_sources = [
            "https://en.wikipedia.org/wiki/Digital_Services_Act",
            "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB976",
            "https://www.flsenate.gov/Session/Bill/2024/3",
            "https://en.wikipedia.org/wiki/Utah_Social_Media_Regulation_Act",
            "https://www.law.cornell.edu/uscode/text/18/2258A",
            "https://gdpr-info.eu/",
            "https://www.privacy-regulation.eu/",
            "https://oag.ca.gov/privacy/ccpa",
            "https://digital-strategy.ec.europa.eu/en/policies/digital-services-act",
            "https://digital-strategy.ec.europa.eu/en/policies/digital-markets-act",
            "https://www.pdpc.gov.sg/Personal-Data-Protection-Act/Overview",
            "https://www.ppc.go.jp/en/",
        ]

        self.glossary = {
            "NR": "Not recommended",
            "PF": "Personalized feed",
            "GH": "Geo-handler; a module responsible for routing features based on user region",
            "CDS": "Compliance Detection System",
            "DRT": "Data retention threshold; duration for which logs can be stored",
            "LCP": "Local compliance policy",
            "Redline": "Flag for legal review (different from its traditional business use for 'financial loss')",
            "Softblock": "A user-level limitation applied silently without notifications",
            "Spanner": "A synthetic name for a rule engine (not to be confused with Google Spanner)",
            "ShadowMode": "Deploy feature in non-user-impact way to collect analytics only",
            "T5": "Tier 5 sensitivity data; more critical than T1–T4 in this internal taxonomy",
            "ASL": "Age-sensitive logic",
            "Glow": "A compliance-flagging status, internally used to indicate geo-based alerts",
            "NSP": "Non-shareable policy (content should not be shared externally)",
            "Jellybean": "Feature name for internal parental control system",
            "EchoTrace": "Log tracing mode to verify compliance routing",
            "BB": "Baseline Behavior; standard user behavior used for anomaly detection",
            "Snowcap": "A synthetic codename for the child safety policy framework",
            "FR": "Feature rollout status",
            "IMT": "Internal monitoring trigger",
        }

        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.feedback_vectorstore = None
        self.retriever = None
        self.feedback_retriever = None
        self.setup_knowledge_base()
        self.setup_feedback_mechanism()

    def _is_feedback_logical(self, original_query: Dict, original_analysis: Dict, user_correction: str) -> bool:
        """Uses the LLM to perform a meta-analysis on the user's feedback."""
        print("🤔 Validating feedback for logical consistency...")

        validation_prompt_text = f"""
        You are a logical validation expert. Your task is to determine if the user's feedback for a compliance analysis is relevant and makes logical sense. Do not judge correctness, only if the reasoning is coherent.

        CONTEXT:
        - Original Feature: {original_query['title']} - {original_query['description']}
        - AI's Original Analysis: The AI decided the compliance flag should be '{original_analysis.get('flag', 'N/A')}' with the reasoning: '{original_analysis.get('reasoning', 'N/A')}'
        - User's Corrective Feedback: '{user_correction}'

        QUESTION:
        Does the user's corrective feedback provide a logical, relevant, and coherent reason for *changing the compliance flag*? For example, if the AI said 'YES' and the user wants to change it, does their reason make sense in a legal or technical compliance context?

        Your response must be a single word: YES or NO.
        """

        response = self.llm.invoke(validation_prompt_text.strip())
        decision = response.content.strip().upper()

        print(f"🕵️ Validation model decision: {decision}")
        return decision == "YES"

    # def load_existing_knowledge_base(self):
    #     """Load existing legal vectorstore from disk"""
    #     print("📖 Loading existing legal knowledge base from cache...")
    #     try:
    #         self.vectorstore = Chroma(
    #             persist_directory="./chroma_db",
    #             embedding_function=OpenAIEmbeddings(),
    #             collection_name="legal_knowledge_base"
    #         )
    #         self.retriever = self.vectorstore.as_retriever(
    #             search_type="mmr", 
    #             search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
    #         )
    #         print("✅ Legal knowledge base loaded from cache.")
    #     except Exception as e:
    #         print(f"❌ Failed to load from cache, rebuilding... Error: {e}")
    #         self.setup_knowledge_base()

    def setup_knowledge_base(self):
        """Build legal knowledge base with structured, law-specific chunking and metadata."""
        print("🔍 Loading legal documents from the web (this may take a few minutes)...")

        law_specific_data = {
            "https://en.wikipedia.org/wiki/Digital_Services_Act": {"law_type": "EU Digital Services Act", "separators": ["== Article", "=== Recitals", "\n\n"]},
            "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB976": {"law_type": "California SB976", "separators": ["SEC.", "\n\n", "\n"]},
            "https://www.flsenate.gov/Session/Bill/2024/3": {"law_type": "Florida Online Protections for Minors Act", "separators": ["Section ", "\n\n", "\n"]},
            "https://en.wikipedia.org/wiki/Utah_Social_Media_Regulation_Act": {"law_type": "Utah Social Media Regulation Act", "separators": ["== Provisions", "=== References", "\n\n"]},
            "https://www.law.cornell.edu/uscode/text/18/2258A": {"law_type": "US Law on Child Sexual Abuse Content", "separators": ["(a)", "(b)", "\n\n"]},
            "https://gdpr-info.eu/": {"law_type": "General Data Protection Regulation (GDPR)", "separators": ["Article ", "Section ", "\n\n"]},
            "https://www.privacy-regulation.eu/": {"law_type": "EU General Data Protection Regulation", "separators": ["Article ", "Section ", "\n\n"]},
            "https://oag.ca.gov/privacy/ccpa": {"law_type": "California Consumer Privacy Act (CCPA)", "separators": ["§", "\n\n", "\n"]},
            "https://digital-strategy.ec.europa.eu/en/policies/digital-services-act": {"law_type": "EU Digital Services Act", "separators": ["Section 1", "Section 2", "\n\n"]},
            "https://digital-strategy.ec.europa.eu/en/policies/digital-markets-act": {"law_type": "EU Digital Markets Act", "separators": ["Article ", "\n\n"]},
            "https://www.pdpc.gov.sg/Personal-Data-Protection-Act/Overview": {"law_type": "Singapore Personal Data Protection Act", "separators": ["Part ", "\n\n"]},
            "https://www.ppc.go.jp/en/": {"law_type": "Japanese Act on the Protection of Personal Information", "separators": ["Chapter ", "Article ", "\n\n"]},
        }

        cache_dir = "legal_docs_cache"
        os.makedirs(cache_dir, exist_ok=True)

        chunks_with_metadata = []

        CACHE_EXPIRATION_DAYS = 30
        now = datetime.now()

        for url, data in law_specific_data.items():
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{url_hash}.json")
            cache_is_valid = False

            # Use cached data if available
            if os.path.exists(cache_path):
                # Check if the cache file is older than the expiration period
                file_mod_timestamp = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if (now - file_mod_timestamp).days < CACHE_EXPIRATION_DAYS:
                    cache_is_valid = True

                try:
                    with open(cache_path, 'r') as f:
                        cached_docs = json.load(f)

                    for cached_doc in cached_docs:
                        doc = Document(
                            page_content=cached_doc["page_content"],
                            metadata=cached_doc["metadata"]
                        )
                        chunks_with_metadata.append(doc)
                    print(f"📁 Loaded from cache: {url} with {len(cached_docs)} chunks.")
                    continue
                except Exception as e:
                    print(f"⚠️ Failed to load cache for {url}: {e}")
                    cache_is_valid = False

            # If not cached, fetch and process
            if not cache_is_valid:
              try:
                  print(f"🌐 Fetching: {url}")
                  loader = WebBaseLoader(
                      web_paths=[url],
                      requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0 (ComplianceBot)"}},
                      bs_kwargs=dict(parse_only=bs4.SoupStrainer("body"))
                  )
                  document = loader.load()[0]

                  splitter = RecursiveCharacterTextSplitter(
                      chunk_size=1500,
                      chunk_overlap=200,
                      separators=data.get("separators", ["\n\n", "\n", " "])
                  )
                  splits = splitter.split_documents([document])

                  # Attach metadata to each chunk
                  for split_doc in splits:
                      split_doc.metadata.update({
                          "law_type": data["law_type"],
                          "source_url": url,
                          "last_updated": datetime.now().isoformat()
                      })

                  chunks_with_metadata.extend(splits)

                  # Cache the processed chunks
                  with open(cache_path, 'w') as f:
                      json.dump([
                          {"page_content": s.page_content, "metadata": s.metadata} for s in splits
                      ], f)

                  print(f"✅ Success: {url} with {len(splits)} chunks.")
              except Exception as e:
                  print(f"❌ Failed {url}: {str(e)}")

        if not chunks_with_metadata:
            raise RuntimeError("No documents were loaded. Please check connectivity.")

        # Create vector DB and persist it to disk
        self.vectorstore = Chroma.from_documents(
            documents=chunks_with_metadata,
            embedding=OpenAIEmbeddings(),
            collection_name="legal_knowledge_base",
            collection_metadata={"hnsw:space": "cosine"}
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )

        print(f"📚 Legal knowledge base built and saved with {len(chunks_with_metadata)} chunks.")

    def setup_feedback_mechanism(self):
        """Initializes the feedback vector store."""
        print("🔍 Initializing feedback mechanism...")
        self.feedback_vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            collection_name="feedback_knowledge_base",
            persist_directory="./feedback_db"
        )
        # Get the current number of documents in the feedback store
        try:
            doc_count = self.feedback_vectorstore._collection.count()
            # Dynamically set k based on available documents
            k_value = min(3, doc_count) if doc_count > 0 else 1
            print(f"📊 Feedback store has {doc_count} documents, setting k={k_value}")
        except Exception as e:
            print(f"⚠️ Could not get document count: {e}")
            k_value = 1
        
        self.feedback_retriever = self.feedback_vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k_value}
        )
        print("✅ Feedback mechanism ready.")
        # self.feedback_retriever = self.feedback_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        # print("✅ Feedback mechanism ready.")

    def add_feedback(self, original_query: Dict, original_analysis: Dict, user_correction: str, is_positive: bool):
        """Adds human feedback to the knowledge base."""
        if is_positive:
            feedback_content = f"""
            POSITIVE FEEDBACK:
            Original Feature: {original_query['title']} - {original_query['description']}
            Original Analysis: {json.dumps(original_analysis, indent=2)}
            User Verdict: This analysis was correct. The reasoning should be reinforced for similar cases.
            """
        else:
            if not self._is_feedback_logical(original_query, original_analysis, user_correction):
                print("❌ Feedback rejected. The provided reasoning does not seem logical or relevant for changing the compliance flag. It will not be saved.")
                return

            print("✅ Feedback was validated as logical.")
            feedback_content = f"""
            CORRECTIVE FEEDBACK:
            Original Feature: {original_query['title']} - {original_query['description']}
            Original Analysis was INCORRECT: {json.dumps(original_analysis, indent=2)}
            User Correction: {user_correction}
            """

        feedback_doc = Document(
            page_content=feedback_content.strip(),
            metadata={"source": "human_feedback", "timestamp": datetime.now().isoformat()}
        )
        self.feedback_vectorstore.add_documents([feedback_doc])
        self.feedback_vectorstore.persist()
        print(f"✅ Feedback for '{original_query['title']}' has been saved.")

    def clear_feedback(self):
        """Clears all feedback by deleting the feedback database."""
        print("🗑️ Clearing all user feedback...")
        try:
            import shutil
            if os.path.exists("./feedback_db"):
                shutil.rmtree("./feedback_db")
            print("✅ Feedback database cleared.")
            
            # Re-initialize
            self.setup_feedback_mechanism()
        except Exception as e:
            # This can happen if the collection doesn't exist yet
            print(f"⚠️ Could not clear feedback (this is normal if it was already empty): {e}")
            # Ensure a clean state by running setup again
            self.setup_feedback_mechanism()

    def format_docs(self, docs):
        return "\n\n".join(f"SOURCE: {doc.metadata.get('source', 'Unknown')}\nCONTENT: {doc.page_content}" for doc in docs)
    
    def translate_query(self, title, description, max_subqueries):
        prompt = f"""
You are a helpful assistant that is professional in legal compliance logic and generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
Return a JSON object with fields:
- original: the original short query (one-liner)
- step_back: a broader conceptual question capturing background context
- subqueries: up to {max_subqueries} concise sub-queries that focus on different facets (short sentences)
Feature Title: {title}
Feature Description: {description}
Only return valid JSON.
"""
        resp = self.llm.call_as_llm(prompt) if hasattr(self.llm, "call_as_llm") else self.llm.generate([{"role":"user","content":prompt}])
        # adapt depending on your ChatOpenAI return format
        text = resp[0].text if isinstance(resp, list) else getattr(resp, "content", None) or str(resp)
        # best-effort parse
        try:
            obj = json.loads(text.strip())
        except Exception:
            # fall back to simple format parsing: keep original + step-back tutor
            return [f"{title}: {description}", f"High level principles relevant to {title}", f"{description}"]

        queries = []
        if obj.get("original"):
            queries.append(obj["original"])
        if obj.get("step_back"):
            queries.append(obj["step_back"])
        for q in obj.get("subqueries", []):
            queries.append(q)
        # dedupe & preserve order
        seen = set()
        out = []
        for q in queries:
            s = q.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def retrieve_for_queries(self, metadata_filter, queries, per_query_k):
        results = []
        retriever_with_filter = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "filter": metadata_filter if metadata_filter else None
            }
        )
        for q in queries:
            docs = retriever_with_filter.invoke(q)  # your retriever returns ordered docs
            # note: some retrievers include scores in metadata, others don't.
            # If your retriever stores scores, extract them; else approximate by position.
            formatted = []
            for rank, d in enumerate(docs[:per_query_k], start=1):
                # try to get an explicit score
                score = None
                if hasattr(d, "metadata") and isinstance(d.metadata, dict):
                    score = d.metadata.get("score") or d.metadata.get("similarity_score")
                formatted.append({"doc": d, "score": score, "rank": rank})
            results.append((q, formatted))
        return results

    def fuse_rankings_rrf(self, per_query_results, k: int = 60):
        fused = {}
        for (q, lst) in per_query_results:
            for item in lst:
                doc = item["doc"]
                doc_id = doc.metadata.get("source", None) or getattr(doc, "id", None) or hash(doc.page_content)
                rank = item["rank"]
                # rrf contribution:
                contrib = 1.0 / (k + rank)
                entry = fused.get(doc_id)
                if not entry:
                    entry = {"doc": doc, "fused_score": 0.0, "best_rank": rank, "sources": {q: rank}}
                    fused[doc_id] = entry
                entry["fused_score"] += contrib
                if rank < entry["best_rank"]:
                    entry["best_rank"] = rank
                entry["sources"][q] = rank

        # convert to sorted list (desc fused_score)
        out = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
        return out

    def enhance_query(self, title: str, description: str) -> str:
        """Enhance the query with compliance-specific terms and concepts"""
        # Extract potential countries/regions mentioned
        country_pattern = r'\b(US|USA|United States|UK|United Kingdom|EU|European Union|France|Germany|Brazil|China|India|Japan|Australia|Canada|Mexico)\b'
        countries = re.findall(country_pattern, f"{title} {description}", re.IGNORECASE)

        # Extract potential legal terms
        legal_terms = []
        legal_patterns = [
            r'\b(law|act|regulation|directive|compliance|legal|statute)\b',
            r'\b(GDPR|CCPA|LGPD|PIPL|DPA|HIPAA|COPPA)\b',
            r'\b(privacy|data protection|consumer protection|age verification|content moderation)\b'
        ]

        for pattern in legal_patterns:
            legal_terms.extend(re.findall(pattern, f"{title} {description}", re.IGNORECASE))

        # Build enhanced query
        enhanced_terms = []

        if countries:
            enhanced_terms.extend(countries)
        if legal_terms:
            enhanced_terms.extend(legal_terms)

        # Add compliance-specific terms
        enhanced_terms.extend(["compliance", "regulation", "legal requirement", "geo-specific"])

        enhanced_query = f"{' '.join(set(enhanced_terms))}"
        return enhanced_query

    def _infer_law_types(self, description: str) -> List[str]:
        """
        Uses an LLM to infer relevant law types from a feature description.
        Returns a list of inferred law types or None.
        """
        prompt_template = """
You are a brilliant compliance researcher. Your task is to analyze a feature description and identify the most likely relevant laws or legal frameworks mentioned or implied. Your response will be used to filter a legal knowledge base, so accuracy is critical.

Respond with a single JSON object with a key 'laws' as a list of strings. Each string must be one of the following exact names:
"EU Digital Services Act"
"California SB976"
"Florida Online Protections for Minors Act"
"Utah Social Media Regulation Act"
"US Law on Child Sexual Abuse Content"
"General Data Protection Regulation (GDPR)"
"EU General Data Protection Regulation"
"California Consumer Privacy Act (CCPA)"
"EU Digital Markets Act"
"Singapore Personal Data Protection Act"
"Japanese Act on the Protection of Personal Information"

If a law is implied by a region (e.g., 'Germany' implies 'EU Digital Services Act' or 'GDPR'), include it. If no specific law is identified, return an empty list.

Example:
Description: "The system will implement a curfew-based login restriction for minors in Berlin."
Response: {{"laws": ["EU Digital Services Act", "General Data Protection Regulation (GDPR)"]}}

Example 2:
Description: "Content moderation for all users in San Diego."
Response: {{"laws": ["California SB976"]}}

Feature Description: {description}
Response:
"""
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm
            response = chain.invoke({"description": description})

            inferred_laws_json = json.loads(response.content.strip())
            return inferred_laws_json.get("laws")
        except Exception as e:
            print(f"⚠️ Error during law type inference: {e}")
            return None

    def create_compliance_prompt(self):
        response_schemas = [
            ResponseSchema(name="flag", description="YES, NO, or UNCLEAR"),
            ResponseSchema(name="reasoning", description="Reasoning for decision"),
            ResponseSchema(name="related_regulations", description="List of relevant regulations"),
            ResponseSchema(name="risk_level", description="HIGH, MEDIUM, LOW, or NONE"),
            ResponseSchema(name="recommended_actions", description="Compliance actions or 'None'"),
            ResponseSchema(name="audit_trail", description="Evidence points for audit"),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        template = """You are a senior compliance analyst at TikTok. Your task is to analyze feature artifacts to determine if they require GEO-SPECIFIC COMPLIANCE LOGIC.

**CRITICAL DISTINCTION:**
- ✅ GEO-SPECIFIC COMPLIANCE LOGIC: The feature implements different behavior in specific regions to comply with LOCAL LAWS OR REGULATIONS.
- ❌ BUSINESS LOGIC: The feature has different behavior for business reasons (market testing, phased rollouts, etc.).
- ❓ UNCLEAR: The feature description doesn't specify the intention for geo-specific logic, requiring human evaluation.

**CONTEXT FROM LEGAL KNOWLEDGE BASE:**
{context}

**FEEDBACK FROM USER (PRIORITIZE THIS IF RELEVANT):**
{feedback_context}

**INTERNAL GLOSSARY:**
{glossary}

**FEATURE ARTIFACT TO ANALYZE:**
Title: {title}
Description: {description}

**ANALYSIS GUIDELINES:**
1. Focus on identifying mentions of specific countries, regions, or laws
2. Determine if the regional logic is driven by legal requirements vs business decisions
3. If intention is unclear, flag as UNCLEAR and specify what information is missing
4. Reference specific regulations and articles/sections when possible
5. Assess risk level based on potential regulatory impact
6. PRIORITIZE FEEDBACK CONTEXT over other contexts if it exists and is relevant

**OUTPUT FORMAT:**
{format_instructions}

**EXAMPLES FOR REFERENCE:**
Example 1:
Input: "Feature reads user location to enforce France's copyright rules (download blocking)"
Output: {{
  "flag": "YES",
  "reasoning": "This feature implements geo-specific compliance logic to adhere to France's copyright law. By reading user location and blocking downloads of copyrighted media within France, it ensures that the platform respects national copyright enforcement. The design aligns with the obligations of digital service providers under EU and French copyright regulations, including the EU Copyright Directive (Article 17) and its local implementation. Non-compliance could result in significant penalties, including fines or restrictions on service operations.",
  "related_regulations": ["EU Copyright Directive Article 17", "French implementation of EU Copyright Directive"],
  "risk_level": "HIGH",
  "recommended_actions": "Ensure proper notice to users, implement appeal process, maintain detailed logs",
  "audit_trail": "France-specific download blocking logic; user notification system; appeal process documentation"
}}

Example 2:
Input: "Geofences feature rollout in US for market testing"
Output: {{
  "flag": "NO",
  "reasoning": "This feature implements geofencing for business purposes—specifically, market testing and controlled feature rollout in selected US states—not for legal compliance. While the system restricts access based on location, there is no statutory requirement or regulation mandating such restrictions. The logic is driven entirely by internal business goals, such as controlling beta access, and does not address any local, state, or federal compliance obligation.",
  "related_regulations": [],
  "risk_level": "NONE",
  "recommended_actions": "None",
  "audit_trail": "Business requirement documents showing market testing purpose"
}}

Now analyze the provided feature artifact:"""

        return template, output_parser

    def analyze_feature(self, title: str, description: str, use_feedback: bool = True) -> Dict[str, Any]:
        query_text = f"{title}: {description}"
        
         # 1) translate into multiple queries
        queries = self.translate_query(title, description, max_subqueries=3)
        # ensure original present
        if query_text not in queries:
            queries.insert(0, query_text)


        # 2) add enhanced terms
        enhanced_terms = self.enhance_query(title, description);
        queries = [f"{q} {enhanced_terms}" for q in queries]

        # 3) retrieve per query
        inferred_laws = self._infer_law_types(description)
        metadata_filter = {}
        if inferred_laws:
            metadata_filter = {"law_type": {"$in": inferred_laws}}
            print(f"⚙️ Inferred laws for filtering: {inferred_laws}")
        per_query_results = self.retrieve_for_queries(metadata_filter, queries, per_query_k=10)

        # 4) fuse with RRF
        fused = self.fuse_rankings_rrf(per_query_results, k=60)

         # 5) choose top K unique docs for context
        top_k = 8
        selected_docs = [c["doc"] for c in fused[:top_k]]
        context = self.format_docs(selected_docs)

         # Retrieve feedback vector stores
        query_text_feedback = f"Feature: {title}. Description: {description}"
        try:
            relevant_feedback_docs = self.feedback_retriever.get_relevant_documents(query_text_feedback)
            feedback_context = self.format_docs(relevant_feedback_docs) if relevant_feedback_docs else "No relevant feedback found."
        except Exception as e:
            print(f"⚠️ Error retrieving feedback: {e}")
            feedback_context = "No feedback available due to retrieval error."


        template, output_parser = self.create_compliance_prompt()
        glossary_text = "\n".join([f"{k}: {v}" for k, v in self.glossary.items()])

        prompt = ChatPromptTemplate.from_template(template).partial(
            format_instructions=output_parser.get_format_instructions(),
            glossary=glossary_text
        )

        chain = (
            {
                "context": lambda x: context,
                "feedback_context": lambda x: feedback_context,
                "title": lambda x: title,
                "description": lambda x: description
            }
            | prompt
            | self.llm
            | output_parser
        )

        result = chain.invoke({})
        result["feature_title"] = title
        result["feature_description"] = description
        result["analysis_timestamp"] = datetime.now().isoformat()
        result["context_sources"] = [doc.metadata.get("source", "Unknown") for doc in selected_docs]
        result["fused_ranking"] = [
            {"source": c["doc"].metadata.get("source", "Unknown"),
             "fused_score": c["fused_score"],
             "best_rank": c["best_rank"],
             "from_queries": list(c["sources"].keys())}
            for c in fused[:top_k]
        ]
        result["feedback_sources_used"] = [doc.page_content for doc in relevant_feedback_docs]
        return result

    def batch_analyze_features(self, features_list: List[Dict]) -> pd.DataFrame:
        results = []
        for i, feature in enumerate(features_list):
            try:
                print(f"Analyzing feature {i+1}/{len(features_list)}: {feature['title']}")
                results.append(self.analyze_feature(feature["title"], feature["description"]))
            except Exception as e:
                print(f"❌ Error analyzing {feature.get('title', 'Unknown')}: {e}")
                # Add error result
                results.append({
                    "flag": "ERROR",
                    "reasoning": f"Analysis failed: {str(e)}",
                    "related_regulations": [],
                    "risk_level": "UNKNOWN",
                    "recommended_actions": "Please try again or check the input format",
                    "feature_title": feature.get("title", "Unknown"),
                    "feature_description": feature.get("description", ""),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "context_sources": []
                })
        return pd.DataFrame(results)

    def display_analysis(self, result: Dict[str, Any]):
        """Display analysis results in a clean format"""
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        print(f"📍 FEATURE: {result['feature_title']}")
        print("")
        
        if result['flag'] == 'YES':
            print("✅ REQUIRES GEO-COMPLIANCE LOGIC")
            print("   This feature needs region-specific implementation")
        elif result['flag'] == 'NO':
            print("❌ NO GEO-COMPLIANCE NEEDED") 
            print("   This feature doesn't require special regional handling")
        else:
            print("⚠️  NEEDS MANUAL REVIEW")
            print("   This feature should be reviewed by a compliance expert")
        
        print("")
        print(f"📈 Risk Level: {result['risk_level']}")
        
        print(f"\n🧠 Why this matters:")
        reasoning = result['reasoning']
        words = reasoning.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > 70:
                lines.append(current_line)
                current_line = word
            else:
                current_line += " " + word if current_line else word
        if current_line:
            lines.append(current_line)
        for line in lines:
            print(f"   {line}")
        
        print(f"\n📚 Related regulations:")
        if isinstance(result['related_regulations'], list):
            for reg in result['related_regulations']:
                print(f"   • {reg}")
        else:
            print(f"   • {result['related_regulations']}")
        
        print(f"\n💡 Next steps:")
        print(f"   {result['recommended_actions']}")
        
        if result.get("inferred_laws"):
            print(f"\n🔍 Inferred relevant laws: {', '.join(result['inferred_laws'])}")
        
        if result.get("feedback_used"):
            print(f"\n🔍 Feedback used: Yes (learned from previous corrections)")

# Mode 1: Interactive single feature analysis with feedback
def run_interactive_analysis():
    print("="*60)
    print("🛡️  COMPLIANCE GUARDIAN - Interactive Learning Mode")
    print("="*60)
    print("This system learns from your feedback and validates it for logical consistency!")
    print("Type 'exit' at any time to quit.")
    print("Type 'clear' to reset feedback learning.")
    print("="*60)
    
    # Initialize the system
    analyzer = ComplianceGuardian(use_cache=True)
    analyzer.clear_feedback()
    while True:
        try:
            # Get feature title
            print("\n💡 What's the name of the feature you want to analyze?")
            title = input("> Feature name: ").strip()
            
            if title.lower() == 'exit':
                print("👋 Goodbye!")
                break
            if title.lower() == 'clear':
                analyzer.clear_feedback()
                print("🧹 Feedback history cleared!")
                continue
            if not title:
                print("⚠️  Please enter a feature name.")
                continue
                
            # Get feature description
            print(f"\n📝 Tell me about '{title}':")
            description = input("> Description: ").strip()
            
            if description.lower() == 'exit':
                print("👋 Goodbye!")
                break
            if not description:
                print("⚠️  Please describe the feature.")
                continue
            
            # Analyze the feature
            print(f"\n🔎 Analyzing '{title}'...")
            result = analyzer.analyze_feature(title, description, use_feedback=True)
            
            # Display results
            analyzer.display_analysis(result)
            
            # Collect feedback
            print("\n" + "-"*60)
            print("💬 Was this analysis correct? (yes/no/skip)")
            feedback = input("> ").strip().lower()
            
            if feedback == 'exit':
                print("👋 Goodbye!")
                break
            elif feedback == 'yes':
                analyzer.add_feedback(
                    {"title": title, "description": description},
                    result,
                    "Analysis confirmed correct",
                    True
                )
                print("✅ Thank you! I'll reinforce this pattern.")
            elif feedback == 'no':
                print("📝 What should the correct analysis be?")
                print("   Example: 'Should be YES because of California law XYZ'")
                correction = input("> Correction: ").strip()
                if correction and correction.lower() != 'exit':
                    analyzer.add_feedback(
                        {"title": title, "description": description},
                        result,
                        correction,
                        False
                    )
            
            print("\n" + "-"*60)
            print("✨ Ready for another feature analysis!")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Sorry, something went wrong: {e}")
            print("Let's try again...")

# Mode 2: Batch CSV processing (no feedback)
def run_batch_analysis():
    print("="*60)
    print("📊 COMPLIANCE GUARDIAN - Batch Processing Mode")
    print("="*60)
    print("This mode processes multiple features from a CSV file.")
    print("Please make sure your CSV file is in the current folder.")
    print("CSV format: feature_title, feature_description")
    print("="*60)
    
    # Initialize the system
    analyzer = ComplianceGuardian(use_cache=True)
    
    # List available CSV files
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in the current directory.")
        print("Please add a CSV file and try again.")
        return
    
    print("\n📁 Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"   {i}. {file}")
    
    # Get file selection
    try:
        selection = input("\nEnter the number of the CSV file to analyze: ").strip()
        if selection.lower() == 'exit':
            return
            
        selected_index = int(selection) - 1
        if selected_index < 0 or selected_index >= len(csv_files):
            print("❌ Invalid selection.")
            return
            
        csv_filename = csv_files[selected_index]
        print(f"📖 Reading file: {csv_filename}")
        
        # Read CSV file
        features = []
        with open(csv_filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'feature_title' in row and 'feature_description' in row:
                    features.append({
                        'title': row['feature_title'],
                        'description': row['feature_description']
                    })
                # Support alternative column names
                elif len(row) >= 2:
                    keys = list(row.keys())
                    features.append({
                        'title': row[keys[0]],
                        'description': row[keys[1]]
                    })
        
        if not features:
            print("❌ No valid features found in the CSV file.")
            print("Please check that your CSV has 'feature_title' and 'feature_description' columns.")
            return
        
        print(f"✅ Found {len(features)} features to analyze.")
        
        # Process batch
        results_df = analyzer.batch_analyze_features(features)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compliance_batch_results_{timestamp}.csv"
        results_df.to_csv(output_filename, index=False)
        
        print(f"\n💾 Results saved to: {output_filename}")
        print(f"📊 Summary:")
        print(f"   Total features processed: {len(results_df)}")
        print(f"   Requires compliance: {len(results_df[results_df['flag'] == 'YES'])}")
        print(f"   No compliance needed: {len(results_df[results_df['flag'] == 'NO'])}")
        print(f"   Needs review: {len(results_df[results_df['flag'] == 'UNCLEAR'])}")
        print(f"   Errors: {len(results_df[results_df['flag'] == 'ERROR'])}")
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")

# Main menu
def main():
    print("="*60)
    print("🛡️  COMPLIANCE GUARDIAN")
    print("="*60)
    print("Choose analysis mode:")
    print("1. Interactive Single Feature Analysis (with feedback learning)")
    print("2. Batch CSV Processing (multiple features, no feedback)")
    print("3. Exit")
    print("="*60)
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            run_interactive_analysis()
            break
        elif choice == '2':
            run_batch_analysis()
            break
        elif choice == '3' or choice.lower() == 'exit':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

