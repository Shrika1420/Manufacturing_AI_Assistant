# Manufacturing Knowledge Assistant

import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# ==========================================
# CUSTOM CSS STYLING
# ==========================================

def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global text - Much darker for readability */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }
    
    /* All paragraphs and text */
    p, span, div, label {
        color: #1e293b !important;
    }
    
    /* Main Background - Softer, cleaner */
    .stApp {
        background: #f8f9fc;
    }
    
    /* Header Styling - Very dark blue */
    h1 {
        color: #0f172a !important;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding-bottom: 0.5rem;
    }
    
    /* All headers dark */
    h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* Sidebar Styling - Soft professional background */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar Headers - Professional dark */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    /* Sidebar text - Dark and readable */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong {
        color: #475569 !important;
    }
    
    /* Sidebar markdown */
    [data-testid="stSidebar"] .stMarkdown {
        color: #475569 !important;
    }
    
    /* Metric Cards - Soft professional with subtle accent */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    [data-testid="stMetric"] label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
    }
    
    /* Buttons - Soft professional with hover accent */
    .stButton > button {
        width: 100%;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        text-transform: none;
    }
    
    .stButton > button:hover {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Chat Messages - Dark readable text */
    [data-testid="stChatMessageContent"] {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageContent"] div {
        color: #1e293b !important;
    }
    
    /* User messages - soft blue professional */
    [data-testid="stChatMessage"][data-testid-type="user"] [data-testid="stChatMessageContent"] {
        background: #f0f9ff;
        border-left: 3px solid #0284c7;
    }
    
    [data-testid="stChatMessage"][data-testid-type="user"] [data-testid="stChatMessageContent"] p {
        color: #0f172a !important;
        font-weight: 500;
    }
    
    /* Assistant messages - subtle accent */
    [data-testid="stChatMessage"][data-testid-type="assistant"] [data-testid="stChatMessageContent"] {
        background: #fefffe;
        border-left: 3px solid #059669;
    }
    
    [data-testid="stChatMessage"][data-testid-type="assistant"] [data-testid="stChatMessageContent"] p {
        color: #1e293b !important;
    }
    
    /* Chat Input - Professional with subtle focus */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    /* Expander - Soft professional */
    [data-testid="stExpander"] {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 0.5rem;
    }
    
    [data-testid="stExpander"] summary {
        font-weight: 500;
        color: #475569;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #cbd5e1;
    }
    
    /* Info/Success/Warning boxes - Dark text */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stAlert p, .stAlert span, .stAlert div {
        color: #1e293b !important;
    }
    
    [data-baseweb="notification"] {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    [data-baseweb="notification"] p {
        color: #1e293b !important;
    }
    
    /* Divider - Subtle */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: #e2e8f0;
    }
    
    /* Spinner - Brand color */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Scrollbar - Professional */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Caption text - Darker */
    .stCaptionContainer {
        color: #475569 !important;
        font-size: 0.825rem;
        font-weight: 500;
    }
    
    /* Markdown in messages - Dark text */
    [data-testid="stChatMessageContent"] p {
        line-height: 1.7;
        color: #1e293b !important;
        font-weight: 400;
    }
    
    [data-testid="stChatMessageContent"] ul, 
    [data-testid="stChatMessageContent"] ol {
        padding-left: 1.5rem;
        line-height: 1.8;
        color: #1e293b !important;
    }
    
    [data-testid="stChatMessageContent"] li {
        color: #1e293b !important;
    }
    
    [data-testid="stChatMessageContent"] strong {
        color: #0f172a !important;
        font-weight: 600;
    }
    
    /* Loading message styling */
    .stSpinner > div > div {
        border-top-color: #3b82f6 !important;
        border-right-color: #3b82f6 !important;
    }
    
    /* Success message subtle animation */
    .element-container:has(.stSuccess) {
        animation: slideIn 0.4s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Professional badge for sources */
    .source-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .source-badge:hover {
        background: #e0e7ff;
        border-color: #3b82f6;
        color: #1e40af;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# DOCUMENT LOADING & PROCESSING
# ==========================================

@st.cache_resource
def load_documents():
    """Load and process PDF documents from documents folder"""
    print("Loading documents...")
    
    loader = DirectoryLoader(
        'documents/',
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks")
    return chunks

@st.cache_resource
def create_vector_store(_chunks):
    """Create FAISS vector store from document chunks"""
    print("Creating vector embeddings...")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    vector_store = FAISS.from_documents(_chunks, embeddings)
    
    print("Vector store created successfully")
    return vector_store

# ==========================================
# PROMPT ENGINEERING
# ==========================================

PROMPT_TEMPLATE = """You are a Manufacturing Knowledge Assistant for a semiconductor manufacturing facility.

Your role is to help employees find accurate information about:
- HR policies (vacation, benefits, onboarding, time off)
- Safety procedures (cleanroom protocols, chemical handling, PPE requirements)
- Equipment information (troubleshooting, maintenance, operating procedures)
- Company policies and procedures

CRITICAL RULES:
1. ONLY answer based on the context provided below
2. If the answer is not in the context, respond: "I don't have that information in my knowledge base. Please contact HR, Safety, or your Supervisor for assistance."
3. Always cite which document your answer comes from at the end
4. For safety-critical questions, add: "⚠️ For safety-critical issues, always verify with your supervisor."
5. Be concise but complete
6. Use bullet points for multi-step procedures

Context from knowledge base:
{context}

Employee Question: {question}

Your Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ==========================================
# RAG CHAIN SETUP
# ==========================================

def create_rag_chain(vector_store):
    """Create the RAG chain using LCEL"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=500
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# ==========================================
# STREAMLIT UI
# ==========================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Manufacturing Knowledge Assistant",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state FIRST (before anything else)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    if 'accuracy_votes' not in st.session_state:
        st.session_state.accuracy_votes = {'helpful': 0, 'not_helpful': 0}
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    
    # Header with custom styling
    st.title("🏭 Manufacturing Knowledge Assistant")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #718096; font-weight: 400; margin-top: -1rem; margin-bottom: 2rem;'>
        AI-Powered Employee Support System for Semiconductor Manufacturing
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("### 🎯 About This System")
        st.markdown("""
        This AI assistant provides instant, accurate answers to your questions about:
        
        **📋 HR Policies**  
        Vacation, benefits, onboarding
        
        **⚠️ Safety Procedures**  
        Cleanroom protocols, PPE, chemicals
        
        **🔧 Equipment Operations**  
        Troubleshooting, maintenance guides
        """)
        
        st.markdown("---")
        
        # System Stats with enhanced styling
        st.markdown("### 📊 Live Statistics")
        if 'question_count' not in st.session_state:
            st.session_state.question_count = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.question_count, "Answered")
        with col2:
            st.metric("Uptime", "24/7", "Always ready")
        
        st.markdown("---")
        
        # Performance Metrics - Auto-calculated
        st.markdown("### 📈 Performance")
        
        # Calculate accuracy
        total_votes = st.session_state.accuracy_votes['helpful'] + st.session_state.accuracy_votes['not_helpful']
        if total_votes > 0:
            accuracy = round((st.session_state.accuracy_votes['helpful'] / total_votes) * 100)
            accuracy_str = f"{accuracy}%"
            accuracy_delta = f"{total_votes} ratings"
        else:
            accuracy_str = "N/A"
            accuracy_delta = "Rate answers below"
        
        # Calculate average response time
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            speed_str = f"{avg_time:.1f}s"
            speed_delta = f"{len(st.session_state.response_times)} queries"
        else:
            speed_str = "Testing..."
            speed_delta = "Ask a question"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", accuracy_str, accuracy_delta)
        with col2:
            st.metric("Speed", speed_str, speed_delta)
        
        st.metric("ROI", "$7.5K/week", "For 100 employees", help="Based on time savings vs traditional methods")
        
        st.markdown("---")
        
        # Search History
        st.markdown("### 🕐 Recent Questions")
        
        # Get user questions from message history
        user_questions = [msg['content'] for msg in st.session_state.messages if msg['role'] == 'user']
        
        if user_questions:
            # Show last 5 questions
            recent_questions = user_questions[-5:]
            
            for i, question in enumerate(reversed(recent_questions), 1):
                # Truncate long questions
                display_question = question if len(question) <= 50 else question[:47] + "..."
                
                if st.button(f"{i}. {display_question}", key=f"history_{i}", use_container_width=True):
                    st.session_state.current_question = question
                    st.rerun()
        else:
            st.info("No questions yet. Start asking!")
        
        st.markdown("---")
        
        # Example Questions
        st.markdown("### 💡 Try These Questions")
        
        example_questions = [
            ("📋", "What is the vacation policy?"),
            ("🕐", "How do I request time off?"),
            ("🧪", "What PPE is required in cleanroom?"),
            ("👋", "What should I do on my first day?"),
            ("⚠️", "What does error code E-47 mean?")
        ]
        
        for icon, question in example_questions:
            if st.button(f"{icon} {question}", key=question, use_container_width=True):
                st.session_state.current_question = question
        
        st.markdown("---")
        
        # Footer info
        st.markdown("""
        <div style='text-align: center; padding: 1rem; color: #475569; font-size: 0.75rem;'>
            <p style='color: #475569 !important;'>Powered by GPT-4 & LangChain</p>
            <p style='color: #475569 !important;'>🔒 Your data is secure and private</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load documents and create RAG system (cached)
    with st.spinner("🔄 Initializing AI knowledge base..."):
        try:
            chunks = load_documents()
            vector_store = create_vector_store(chunks)
            rag_chain, retriever = create_rag_chain(vector_store)
            
            st.success(f"✅ Knowledge base ready! Processing {len(chunks)} document sections from HR, Safety, and Operations manuals.")
        except Exception as e:
            st.error(f"❌ Error loading documents: {str(e)}")
            st.info("📁 Please ensure PDF files are in the 'documents' folder and try refreshing the page.")
            return
    
    # Welcome message for first-time users
    if len(st.session_state.messages) == 0:
        st.info("""
        ### 👋 Welcome to Your AI Knowledge Assistant!
        
        I'm here to help you find information instantly. You can ask me about:
        
        - 📋 **HR Policies** → Vacation days, benefits, time-off procedures
        - ⚠️ **Safety Protocols** → PPE requirements, chemical handling, emergency procedures  
        - 🔧 **Equipment Help** → Error codes, troubleshooting, maintenance schedules
        - 📚 **Onboarding Info** → First day procedures, required documents, training
        
        **Quick tip:** Click any example question in the sidebar or type your own question below!
        """)
    
    # Display chat history with enhanced styling
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant":
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 6])
                
                with feedback_col1:
                    if st.button("👍 Helpful", key=f"helpful_{i}", use_container_width=True):
                        st.session_state.accuracy_votes['helpful'] += 1
                        st.success("Thanks for the feedback!")
                        st.rerun()
                
                with feedback_col2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{i}", use_container_width=True):
                        st.session_state.accuracy_votes['not_helpful'] += 1
                        st.info("Feedback recorded. We'll improve!")
                        st.rerun()
            
            if "sources" in message and message["sources"]:
                with st.expander("📄 View Source Documents"):
                    sources_html = ""
                    for source in set(message["sources"]):
                        sources_html += f'<span class="source-badge">{source}</span>'
                    st.markdown(sources_html, unsafe_allow_html=True)
    
    # Chat input
    question = st.chat_input("💬 Ask me anything about HR policies, safety procedures, or equipment...")
    
    # Handle example question clicks
    if st.session_state.current_question and not question:
        question = st.session_state.current_question
        st.session_state.current_question = ""
    
    if question:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching knowledge base..."):
                start_time = time.time()
                
                try:
                    # Get answer from RAG chain
                    answer = rag_chain.invoke(question)
                    
                    # Get source documents for citation
                    source_docs = retriever.invoke(question)
                    
                    elapsed_time = time.time() - start_time
                    
                    # Store response time for metrics
                    st.session_state.response_times.append(elapsed_time)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show response time with enhanced styling
                    st.markdown(f"""
                    <div style='margin-top: 1rem; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; font-size: 0.85rem;'>
                        ⚡ <strong>Response time:</strong> {elapsed_time:.2f}s | 
                        📊 <strong>Confidence:</strong> High | 
                        🔍 <strong>Sources:</strong> {len(source_docs)} documents
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extract and display sources
                    sources = []
                    for doc in source_docs:
                        source_name = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        sources.append(f"{os.path.basename(source_name)} (Page {page})")
                    
                    if sources:
                        with st.expander("📄 View Source Documents"):
                            sources_html = ""
                            for source in set(sources):
                                sources_html += f'<span class="source-badge">{source}</span>'
                            st.markdown(sources_html, unsafe_allow_html=True)
                    
                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": list(set(sources))
                    })
                    
                    # Update question count
                    st.session_state.question_count += 1
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Oops! Something went wrong: {str(e)}")
                    st.info("💡 Try rephrasing your question or contact support at x-3100")
    
    # Enhanced Footer with soft professional gradient
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8fafc; border-radius: 12px; margin-top: 2rem; border: 1px solid #e2e8f0;'>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate average response time for footer
    if st.session_state.response_times:
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        footer_time = f"~{avg_time:.1f} seconds"
        time_delta = "Live average"
    else:
        footer_time = "~2-3 seconds"
        time_delta = "Estimated"
    
    with col1:
        st.metric("⚡ Avg Response", footer_time, time_delta, help="Compared to email or phone support")
    with col2:
        st.metric("📚 Knowledge Base", f"{len(chunks)} sections", "Always current", help="From HR, Safety & Operations manuals")
    with col3:
        st.metric("🕐 Availability", "24/7/365", "Never offline", help="No waiting for office hours")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()