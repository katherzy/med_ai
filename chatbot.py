import os
import io
import streamlit as st
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, END

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

# Optional PDF reader
try:
    from pypdf import PdfReader
except:
    PdfReader = None

# Global Constants
DB_FAISS_PATH = "vector_db/db_faiss"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
LLM_TEMPERATURE = 0.0
RETRIEVER_K = 3
MAX_RECORDS_SNIPPET_LENGTH = 1000

INITIAL_BOT_MESSAGE = (
    "Hello — I'm a symptom triage assistant. I can use your previous medical records "
    "to personalize recommendations. Upload records (PDF / TXT) below if you'd like."
)

TRIAGE_QUESTIONS = [
    ("primary_symptom", "What is the main symptom you are experiencing? (e.g. chest pain, fever, cough)"),
    ("severity", "How severe are the symptoms? (mild/moderate/severe or 1-10)"),
    ("duration", "How long have you had this symptom? (hours / days)"),
    ("age", "How old is the patient?"),
    ("other", "Any other symptoms or medical conditions? (pregnancy, immunocompromised, blood thinners, etc.)"),
]

SEVERE_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath", "unconscious",
    "loss of consciousness", "stroke", "slurred speech", "sudden weakness",
    "severe bleeding", "not breathing"
]

URGENT_KEYWORDS = [
    "high fever", "fever", "persistent vomiting", "severe pain", "deep cut",
    "possible fracture", "infected", "worsening", "dehydration"
]

RECORD_ESCALATORS = [
    "heart disease", "myocardial", "anticoagulant", "blood thinner",
    "immunocompromised", "chemotherapy", "pregnant", "organ transplant",
    "arrhythmia", "stroke history"
]

# Lists based on ccmedicalcenter.com article: Top reasons people visit the ER
MENS_TREAT_AND_RELEASE = [
    "open wounds to the head",
    "open wounds to the neck",
    "open wounds to the limbs",
    "head injury",
    "neck injury",
    "limb injury"
]

WOMENS_TREAT_AND_RELEASE = [
    "urinary tract infection",
    "UTI",
    "headache",
    "migraine",
    "pregnancy-related issues",
    "pregnancy complications"
]

ER = [
    "allergic reactions with trouble breathing",
    "severe swelling from allergic reaction",
    "broken bones",
    "dislocations",
    "loss of vision",
    "double vision",
    "choking",
    "electric shock",
    "severe head injury",
    "heart attack symptoms",
    "chest pain with shortness of breath",
    "high fever over 103F",
    "fever with rash",
    "loss of consciousness",
    "mental health crisis",
    "self-harm",
    "harm to others",
    "poisoning",
    "seizures",
    "severe abdominal pain"
]

URGENT_CARE = [
    "minor cuts",
    "minor wounds",
    "superficial injuries",
    "mild fever",
    "upper respiratory infection",
    "bronchiolitis",
    "mild asthma attack",
    "middle ear infection",
    "viral infection",
    "minor musculoskeletal pain",
    "minor back pain",
    "non-severe headache",
    "mild abdominal pain",
    "vomiting without severe dehydration"
]

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

SEVERE_TRIAGE_THRESHOLD = 8
MILD_TRIAGE_THRESHOLD = 4
YOUNG_AGE_THRESHOLD = 2
SENIOR_AGE_THRESHOLD = 75

ER_RECOMMENDATION = (
    "Recommendation: Go to the Emergency Room (ER) now. "
    "If someone is unresponsive or not breathing, call emergency services immediately."
)

URGENT_CARE_RECOMMENDATION = "Recommendation: Seek Urgent Care as soon as possible for evaluation and treatment."

PRIMARY_CARE_RECOMMENDATION = (
    "Recommendation: Contact your primary care provider to schedule an appointment. "
    "If symptoms worsen, seek urgent or emergency care."
)

# ============================================================================
# AGENT STATE DEFINITIONS
# ============================================================================

class AgentState(TypedDict):
    """Main state shared across all agents"""
    task_type: str  # 'triage', 'qa', 'record_analysis'
    user_query: str
    triage_answers: dict
    medical_records_text: str

    # Agent outputs
    triage_result: str
    triage_recommendation: str
    record_analysis: str
    qa_context: str
    qa_answer: str
    source_documents: List

    # Main agent output
    final_response: str
    next_action: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf") and PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""
    else:
        # Try decode as text
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def reset_triage():
    st.session_state.messages = [
        {"role": "assistant", "content": INITIAL_BOT_MESSAGE},
        {"role": "assistant", "content": TRIAGE_QUESTIONS[0][1]},
    ]
    st.session_state.question_index = 0
    st.session_state.answers = {}
    st.session_state.triage_complete = False
    st.session_state.medical_records = []
    st.session_state.medical_records_text = ""

# ============================================================================
# AGENT NODES
# ============================================================================

def triage_agent(state: AgentState) -> AgentState:
    """
    Triage Agent: Evaluates patient symptoms and medical history to determine
    appropriate care level (ER, Urgent Care, or Primary Care)
    """
    answers = state.get("triage_answers", {})
    records_text = state.get("medical_records_text", "")

    primary = answers.get("primary_symptom", "").lower()
    severity = answers.get("severity", "").lower()
    duration = answers.get("duration", "").lower()
    other = answers.get("other", "").lower()
    age = answers.get("age", "").strip()
    records = (records_text or "").lower()

    combined = " ".join([primary, other, records])

    # Check for severe/emergency keywords
    for kw in SEVERE_KEYWORDS:
        if kw in combined:
            state["triage_result"] = "ER"
            state["triage_recommendation"] = ER_RECOMMENDATION
            return state

    # Check record escalators
    for kw in RECORD_ESCALATORS:
        if kw in records:
            if "severe" in severity or (severity.isdigit() and int(severity) >= SEVERE_TRIAGE_THRESHOLD):
                state["triage_result"] = "ER"
                state["triage_recommendation"] = ER_RECOMMENDATION
                return state
            state["triage_result"] = "Urgent Care"
            state["triage_recommendation"] = URGENT_CARE_RECOMMENDATION
            return state

    # Numeric severity check
    try:
        sev_num = int(severity)
    except Exception:
        sev_num = None

    if sev_num is not None and sev_num >= SEVERE_TRIAGE_THRESHOLD:
        state["triage_result"] = "ER"
        state["triage_recommendation"] = ER_RECOMMENDATION
        return state

    if "severe" in severity:
        state["triage_result"] = "ER"
        state["triage_recommendation"] = ER_RECOMMENDATION
        return state

    # Check urgent keywords
    for kw in URGENT_KEYWORDS:
        if kw in combined or kw in severity:
            state["triage_result"] = "Urgent Care"
            state["triage_recommendation"] = URGENT_CARE_RECOMMENDATION
            return state

    # Age-based escalation
    try:
        age_num = int(age)
        if age_num <= YOUNG_AGE_THRESHOLD or age_num >= SENIOR_AGE_THRESHOLD:
            if primary or other or records:
                state["triage_result"] = "Urgent Care"
                state["triage_recommendation"] = URGENT_CARE_RECOMMENDATION
                return state
    except Exception:
        pass

    # Duration-based triage
    if any(word in duration for word in ["day", "days", "week", "weeks"]):
        if "mild" in severity or (sev_num is not None and sev_num <= MILD_TRIAGE_THRESHOLD):
            state["triage_result"] = "Primary Care"
            state["triage_recommendation"] = PRIMARY_CARE_RECOMMENDATION
            return state
        state["triage_result"] = "Urgent Care"
        state["triage_recommendation"] = URGENT_CARE_RECOMMENDATION
        return state

    # Default to primary care
    state["triage_result"] = "Primary Care"
    state["triage_recommendation"] = PRIMARY_CARE_RECOMMENDATION
    return state

def record_analysis_agent(state: AgentState) -> AgentState:
    """
    Record Analysis Agent: Analyzes patient medical records for relevant
    conditions, medications, and risk factors
    """
    records_text = state.get("medical_records_text", "")

    if not records_text:
        state["record_analysis"] = "No medical records provided."
        return state

    # Use LLM to analyze records for relevant conditions
    llm = ChatGroq(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
    )

    analysis_prompt = f"""
    Analyze the following medical records and extract key information relevant to symptom triage:
    - Chronic conditions
    - Current medications (especially blood thinners, immunosuppressants)
    - Risk factors (pregnancy, age, immunocompromised status)
    - Relevant medical history
    
    Medical Records:
    {records_text[:2000]}
    
    Provide a concise summary of relevant findings:
    """

    try:
        response = llm.invoke(analysis_prompt)
        analysis = response.content if hasattr(response, 'content') else str(response)
        state["record_analysis"] = analysis
    except Exception as e:
        state["record_analysis"] = f"Error analyzing records: {str(e)}"

    return state

def qa_agent(state: AgentState) -> AgentState:
    """
    QA Agent: Retrieves relevant medical information from vector database
    and generates answers to user questions
    """
    query = state.get("user_query", "")
    records_text = state.get("medical_records_text", "")

    # Build enhanced query with patient context
    full_query = query
    if records_text:
        full_query = f"Patient records: {records_text[:1000]}\n\nUser question: {query}"

    try:
        # Retrieve documents
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})
        docs = retriever.get_relevant_documents(full_query)

        state["source_documents"] = docs
        context = "\n\n".join([doc.page_content for doc in docs])
        state["qa_context"] = context

        # Generate answer
        llm = ChatGroq(
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        )

        prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        formatted_prompt = prompt.format(context=context, question=query)
        response = llm.invoke(formatted_prompt)

        state["qa_answer"] = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        state["qa_answer"] = f"Error generating answer: {str(e)}"

    return state

def main_agent(state: AgentState) -> AgentState:
    """
    Main Agent: Coordinates sub-agents and coalesces their responses into
    a comprehensive final response
    """
    task_type = state.get("task_type", "")

    # Coalesce responses based on what agents have been called
    final_parts = []

    # If triage was performed
    if state.get("triage_result"):
        final_parts.append("=== TRIAGE ASSESSMENT ===")
        final_parts.append(f"Care Level: {state['triage_result']}")
        final_parts.append(state["triage_recommendation"])
        final_parts.append("")

    # If records were analyzed
    if state.get("record_analysis") and state["record_analysis"] != "No medical records provided.":
        final_parts.append("=== MEDICAL RECORD ANALYSIS ===")
        final_parts.append(state["record_analysis"])
        final_parts.append("")

    # If QA was performed
    if state.get("qa_answer"):
        final_parts.append("=== MEDICAL INFORMATION ===")
        final_parts.append(state["qa_answer"])
        if state.get("source_documents"):
            final_parts.append("\n--- Sources ---")
            for i, doc in enumerate(state["source_documents"][:3], 1):
                final_parts.append(f"{i}. {doc.page_content[:200]}...")
        final_parts.append("")

    # Add summary of triage answers if available
    if state.get("triage_answers") and state.get("triage_result"):
        final_parts.append("=== YOUR RESPONSES ===")
        for k, q in TRIAGE_QUESTIONS:
            answer = state["triage_answers"].get(k, "No answer")
            final_parts.append(f"• {q.split('?')[0]}? → {answer}")

    state["final_response"] = "\n".join(final_parts) if final_parts else "Processing complete."
    state["next_action"] = "complete"

    return state

def route_to_task(state: AgentState) -> str:
    """Router function to determine which agent to call from entry point"""
    task_type = state.get("task_type", "")

    # Route directly to the appropriate specialist agent
    if task_type == "triage":
        return "triage"
    elif task_type == "qa":
        return "qa"
    elif task_type == "record_analysis":
        return "record_analysis"
    else:
        return "complete"

# ============================================================================
# LANGGRAPH WORKFLOW CREATION
# ============================================================================

@st.cache_resource
def create_agent_graph():
    """Create multi-agent LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("triage_agent", triage_agent)
    workflow.add_node("record_analysis_agent", record_analysis_agent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("main_agent", main_agent)

    # Set entry point with conditional routing
    workflow.set_conditional_entry_point(
        route_to_task,
        {
            "triage": "triage_agent",
            "qa": "qa_agent",
            "record_analysis": "record_analysis_agent",
            "complete": "main_agent",
        }
    )

    # All sub-agents route to main agent for final coalescing, then END
    workflow.add_edge("triage_agent", "main_agent")
    workflow.add_edge("record_analysis_agent", "main_agent")
    workflow.add_edge("qa_agent", "main_agent")
    workflow.add_edge("main_agent", END)

    return workflow.compile()

def run_agent_workflow(task_type: str, user_query: str = "", triage_answers: dict = None, medical_records_text: str = ""):
    """Execute the agent workflow with given parameters"""
    agent_graph = create_agent_graph()

    initial_state = AgentState(
        task_type=task_type,
        user_query=user_query,
        triage_answers=triage_answers or {},
        medical_records_text=medical_records_text,
        triage_result="",
        triage_recommendation="",
        record_analysis="",
        qa_context="",
        qa_answer="",
        source_documents=[],
        final_response="",
        next_action=""
    )

    # Add config with increased recursion limit as safety measure
    config = {"recursion_limit": 50}

    final_state = agent_graph.invoke(initial_state, config=config)
    return final_state

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("Multi-Agent Symptom Triage Chatbot")
    st.caption("Powered by LangGraph Multi-Agent Framework")

    if "messages" not in st.session_state:
        reset_triage()

    # Medical records uploader
    uploaded = st.file_uploader(
        "Upload prior medical records (PDF or TXT)\nYou can upload multiple files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded:
        texts: List[str] = []
        for f in uploaded:
            text = extract_text_from_file(f)
            texts.append(f"{f.name}:\n{text}")
            if f.name not in st.session_state.medical_records:
                st.session_state.medical_records.append(f.name)
        combined_text = "\n\n".join(texts)
        st.session_state.medical_records_text = (
            st.session_state.get("medical_records_text", "") + "\n\n" + combined_text
        ).strip()

        # Run record analysis agent
        result = run_agent_workflow(
            task_type="record_analysis",
            medical_records_text=st.session_state.medical_records_text
        )

        msg = f"Medical records uploaded and analyzed.\n\n{result.get('record_analysis', '')}"
        st.chat_message("assistant").markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})

    # Render chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_input = st.chat_input("Type here (or type `restart` to start over)")
    if not user_input:
        return

    if user_input.strip().lower() == "restart":
        reset_triage()
        st.rerun()

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.get("triage_complete", False):
        # After triage, use QA agent
        result = run_agent_workflow(
            task_type="qa",
            user_query=user_input,
            medical_records_text=st.session_state.get("medical_records_text", "")
        )

        response = result.get("final_response", "No response generated.")
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return

    # Collect triage answers
    q_index = st.session_state.get("question_index", 0)
    key = TRIAGE_QUESTIONS[q_index][0]
    st.session_state.answers[key] = user_input.strip()
    q_index += 1
    st.session_state.question_index = q_index

    if q_index < len(TRIAGE_QUESTIONS):
        next_q = TRIAGE_QUESTIONS[q_index][1]
        st.chat_message("assistant").markdown(next_q)
        st.session_state.messages.append({"role": "assistant", "content": next_q})
    else:
        # Run triage agent workflow
        result = run_agent_workflow(
            task_type="triage",
            triage_answers=st.session_state.answers,
            medical_records_text=st.session_state.get("medical_records_text", "")
        )

        final_response = result.get("final_response", "Unable to complete triage.")
        st.chat_message("assistant").markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.session_state.triage_complete = True

if __name__ == "__main__":
    main()
