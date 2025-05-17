import streamlit as st
import configparser
import os
import sys

# プロジェクトルートをパスに追加
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root = get_project_root()
sys.path.append(project_root)

from model_loader.load_llm import load_llm
from components.pdf_processor import PDFProcessor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json

#
import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# ---------- 設定 ----------
config = configparser.ConfigParser()
config.read(os.path.join(project_root, 'DeepRag', 'config.ini'))
OLLAMA_BASE_URL = config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
EMBEDDING_MODEL = config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small')
LLM_MODEL = config.get('llm', 'MODEL', fallback='llama2')
PERSIST_DIRECTORY = config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore')
PDF_PATH = config.get('document', 'PATH', fallback='MRD_Rag/2504.13079v1.pdf')
NUM_AGENTS = config.getint('mrdrag', 'NUM_AGENTS', fallback=4)
MAX_ROUNDS = config.getint('mrdrag', 'MAX_ROUNDS', fallback=10)
NUM_SEARCH_RESULTS = config.getint('vectorstore', 'NUM_SEARCH_RESULTS', fallback=10)


# ---------- プロンプト ----------
AGENT_PROMPT = PromptTemplate(
    input_variables=["query", "document", "agent_id", "role", "agg_summary", "agg_explanation"],
    template="""
You are Agent {agent_id}, serving as a {role} in a multi-agent RAG debate.
Use only the provided document to produce evidence or challenge existing summaries.
If agg_summary is "None", provide your initial stance.
Otherwise, defend, revise, or challenge based on:
Aggregate Summary: {agg_summary}
Explanation: {agg_explanation}
Be factual and concise, focusing on unique insights from the document.

Query: {query}
Document:
{document}

Your response:
"""
)

AGGREGATOR_PROMPT = PromptTemplate(
    input_variables=["query", "agent_summaries"],
    template="""
You are the central Aggregator in a multi-agent debate.
Given agent summaries with diverse perspectives, produce:
1) A final concise answer(s) for ambiguous queries.
2) Explanation of choices, discarding misinformation or noise.

Agent Summaries:
{agent_summaries}

Query: {query}

Respond with JSON:
{{
  "summary": "<final answer>",
  "explanation": "<reasoning>"
}}
"""
)

# ---------- Diversity settings ----------
AGENT_TEMPS = [0.7, 0.5, 0.3, 0.1]  # 各エージェントの温度
AGENT_ROLES = [
    "skeptical critic",
    "detail-oriented summarizer",
    "concise responder",
    "contextual cross-checker"
]

# ---------- MRD-RAGコア ----------
class MRDRagCore:
    # Initialize the MRD-RAG core with LLM and vectorstore
    # num_agents: Number of agents to use
    # max_rounds: Maximum number of rounds for debate
    # num_search_results: Number of search results to retrieve
    # vectorstore: Vector store for document retrieval
    # llm: Language model for agent and aggregator
    #
    # AGENT_TEMPS: List of temperatures for agents
    # AGENT_ROLES: List of roles for agents
    # AGENT_PROMPT: Prompt template for agents
    # AGGREGATOR_PROMPT: Prompt template for aggregator
    def __init__(self, llm, vectorstore, num_agents=4, max_rounds=10, num_search_results=10):
        self.llm = llm
        self.vectorstore = vectorstore
        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.num_search_results = num_search_results

    def run(self, question):
        docs = self.vectorstore.similarity_search(question, k=self.num_search_results)
        agent_summaries = {i+1: None for i in range(self.num_agents)}
        agg_summary = "None"
        agg_explanation = "None"
        round_traces = []

        for round_idx in range(1, self.max_rounds+1):
            agent_outputs = {}
            for idx, doc in enumerate(docs, start=1):
                # 各エージェントごとに異なる温度・役割を割り当て
                role = AGENT_ROLES[(idx-1) % len(AGENT_ROLES)]
                temp = AGENT_TEMPS[(idx-1) % len(AGENT_TEMPS)]
                # LLMを温度ごとに生成
                llm_agent = load_llm(
                    provider="ollama",
                    model=LLM_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temp
                )
                chain = LLMChain(llm=llm_agent, prompt=AGENT_PROMPT)
                summary = chain.run(
                    query=question,
                    document=doc.page_content,
                    agent_id=idx,
                    role=role,
                    agg_summary=agg_summary,
                    agg_explanation=agg_explanation
                )
                agent_summaries[idx] = f"[Agent {idx} ({role})]: {summary.strip()}"
                agent_outputs[idx] = summary

            summaries_text = "\n---\n".join(agent_summaries.values())
            # Aggregatorは温度固定
            llm_agg = load_llm(
                provider="ollama",
                model=LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.2
            )
            agg_chain = LLMChain(llm=llm_agg, prompt=AGGREGATOR_PROMPT)
            output = agg_chain.run(query=question, agent_summaries=summaries_text)
            try:
                parsed = json.loads(output)
                agg_summary = parsed.get("summary")
                agg_explanation = parsed.get("explanation")
            except Exception:
                agg_summary = output
                agg_explanation = "Aggregator output could not be parsed as JSON."

            round_traces.append({
                "round": round_idx,
                "agent_outputs": agent_outputs,
                "agg_summary": agg_summary,
                "agg_explanation": agg_explanation
            })

        return agg_summary, round_traces

# ---------- Streamlit UI ----------
def main():
    st.title("MADAM-RAG: Multi-Round Debate & Aggregation")
    st.info(f"Ollama BASE_URL: {OLLAMA_BASE_URL}")

    processor = PDFProcessor(config_path=os.path.join(project_root, 'DeepRag', 'config.ini'))

    if st.button("Index PDF"):
        with st.spinner("Indexing documents..."):
            store = processor.index_pdfs()
            if store:
                st.success("Indexing completed!")
            else:
                st.error("Indexing failed.")

    store = processor.load_vectorstore()
    if store is None:
        st.warning("Vectorstore not found. Please index PDFs first.")
        return

    llm = load_llm(
        provider="ollama",
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )
    mrdrag = MRDRagCore(llm, store, num_agents=NUM_AGENTS, max_rounds=MAX_ROUNDS, num_search_results=NUM_SEARCH_RESULTS)

    question = st.text_input("Enter your question:")
    if st.button("Run MADAM-RAG") and question:
        with st.spinner("Running multi-agent debate..."):
            agg_summary, traces = mrdrag.run(question)
            for trace in traces:
                st.markdown(f"### Round {trace['round']}")
                for idx, summary in trace["agent_outputs"].items():
                    role = AGENT_ROLES[(int(idx)-1) % len(AGENT_ROLES)]
                    with st.expander(f"Agent {idx} ({role}) - Round {trace['round']}"):
                        st.write(summary)
                st.subheader(f"Aggregator - Round {trace['round']}")
                st.markdown(f"**Summary:** {trace['agg_summary']}")
                st.markdown(f"**Explanation:** {trace['agg_explanation']}")
            st.subheader("Final Answer")
            final_output = None
            parsed_successfully = False
            try:
                # Attempt to find and parse JSON using regex
                import re
                json_match = re.search(r'\{.*?\}', agg_summary, re.DOTALL)
                if json_match:
                    json_string = json_match.group(0)
                    final_output = json.loads(json_string)
                    parsed_successfully = True
                else:
                    # If regex doesn't find a JSON object, try stripping and loading as a fallback
                    final_output = json.loads(agg_summary.strip())
                    parsed_successfully = True # If this succeeds, it was valid JSON after stripping

            except json.JSONDecodeError:
                # JSON decode failed after stripping or regex extraction
                parsed_successfully = False
            except Exception as e:
                # Catch any other unexpected errors during parsing
                st.markdown(f"**Parsing Error:** An unexpected error occurred: {e}")
                parsed_successfully = False

            if parsed_successfully and isinstance(final_output, dict):
                st.markdown(f"**Summary:** {final_output.get('summary')}")
                st.markdown(f"**Explanation:** {final_output.get('explanation')}")
            else:
                # Display raw output if parsing failed
                st.markdown("**Summary:**")
                st.markdown(agg_summary)
                st.markdown("**Explanation:** Aggregator output could not be parsed as JSON.")


if __name__ == "__main__":
    main()
