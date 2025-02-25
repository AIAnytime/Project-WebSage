import streamlit as st
import os
import asyncio
import nest_asyncio
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from streamlit_chat import message as st_message

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
nest_asyncio.apply()

# ---------------------------
# Initialize Session State Variables
# ---------------------------
if "url_submitted" not in st.session_state:
    st.session_state.url_submitted = False
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(layout="wide", page_title="WebSage")
st.title("Project WebSage")

page = st.sidebar.selectbox("Navigation", ["Home", "AI Engine", "Contact"])

if page == "Home":
    st.markdown("""
    ## Welcome to WebSage
    **WebSage** is a cutting-edge RAG Chatbot application that allows you to extract content from any URL, generate detailed summaries, and interact with the content using advanced language models.  
    With options to choose between **Closed Source** (OpenAI) and **Open Source** (Ollama) engines for both summarization and conversation, WebSage gives you the flexibility to explore and deploy the best AI solutions for your needs.
    
    **Features:**
    - **Website Extraction:** Crawl and extract web page content.
    - **Summarization:** Generate detailed summaries of the extracted content.
    - **Embeddings & Retrieval:** Create embeddings with FAISS for intelligent document retrieval.
    - **Chatbot Interface:** Interact with your content via a conversational agent.
    
    Get started by selecting **AI Engine** from the sidebar.
    """)

elif page == "AI Engine":

    # ---------------------------
    # URL Input Form
    # ---------------------------
    with st.form("url_form"):
        url_input = st.text_input("Enter a URL to crawl:")
        submit_url = st.form_submit_button("Submit URL")
        if submit_url and url_input:
            st.session_state.url_submitted = True
            # Reset any previous state
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""

    # ---------------------------
    # If URL has been submitted, divide layout into three columns
    # ---------------------------
    if st.session_state.url_submitted:
        col1, col2, col3 = st.columns(3)

        # ---------------------------
        # Column 1: Website Extraction & Summarization using crawl4ai
        # ---------------------------
        with col1:
            st.header("1. Website Extraction")
            if not st.session_state.extraction_done:
                with st.spinner("Extracting website..."):
                    # Define async crawl function (returns markdown output)
                    async def simple_crawl(url):
                        crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
                        async with AsyncWebCrawler() as crawler:
                            result = await crawler.arun(url=url, config=crawler_run_config)
                            return result.markdown

                    # Run the async crawl (nest_asyncio makes this safe in Streamlit)
                    extracted = asyncio.run(simple_crawl(url_input))
                    st.session_state.extracted_text = extracted
                    st.session_state.extraction_done = True
                st.success("Extraction complete!")

            # Show a preview (first few non-empty lines)
            preview = "\n".join(
                [line for line in st.session_state.extracted_text.splitlines() if line.strip()][:5]
            )
            st.text_area("Extracted Text Preview", preview, height=150)

            # Save the full extracted text as a file and provide a download button.
            st.download_button(
                label="Download Extracted Text",
                data=st.session_state.extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain",
            )

            st.markdown("---")
            st.subheader("Summarize Web Page")
            if st.button("Summarize Web Page", key="summarize_button"):
                with st.spinner("Summarizing..."):
                    summary_prompt_template = """
    You are an AI assistant that is tasked with summarizing a web page.
    Your summary should be detailed and cover all key points mentioned in the web page.
    Below is the extracted content of the web page:
    {content}

    Please provide a comprehensive and detailed summary in Markdown format.
    """
                    summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["content"])
                    prompt_text = summary_prompt.format(content=st.session_state.extracted_text)
                    summarizer = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=1500)
                    summary_response = summarizer(prompt_text)
                    st.session_state.summary = summary_response.content
                st.success("Summarization complete!")

            if st.session_state.summary:
                st.subheader("Summarized Output")
                st.markdown(st.session_state.summary, unsafe_allow_html=False)

        # ---------------------------
        # Column 2: Creating Embeddings with FAISS
        # ---------------------------
        with col2:
            st.header("2. Create Embeddings")
            if st.session_state.extraction_done and not st.session_state.embedding_done:
                if st.button("Create Embeddings"):
                    with st.spinner("Creating embeddings..."):
                        # Save extracted text to a markdown file (output.md)
                        with open("output.md", "w", encoding="utf-8") as f:
                            f.write(st.session_state.extracted_text)

                        # Load the markdown file using UnstructuredMarkdownLoader
                        loader = UnstructuredMarkdownLoader("output.md")
                        data = loader.load()

                        # Split the text into chunks using RecursiveCharacterTextSplitter
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        texts = text_splitter.split_documents(data)

                        # Set your OpenAI API key (using environment variable)
                        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

                        # Create embeddings using OpenAIEmbeddings
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

                        # Build a FAISS vectorstore from the documents
                        vectorstore = FAISS.from_documents(texts, embeddings)

                        # Persist the vectorstore locally (optional)
                        vectorstore.save_local("faiss_index")

                        st.session_state.vectorstore = vectorstore
                        st.session_state.embedding_done = True
                    st.success("Vectors are created!")
            elif st.session_state.embedding_done:
                st.info("Embeddings have been created.")

        # ---------------------------
        # Column 3: Chatbot using streamlit_chat and a Retrieval Chain
        # ---------------------------
        with col3:
            st.header("3. Chat with the Bot")
            if st.session_state.embedding_done:
                # Let the user select the LLM type
                llm_choice = st.radio("Select LLM Type", ("Closed Source", "Open Source"), index=0, key="llm_choice")

                # Setup retrieval-based QA chain using the vectorstore
                vectorstore = st.session_state.vectorstore
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                prompt_template = """
    You are an AI assistant tasked with answering questions based solely
    on the provided context. Your goal is to generate a comprehensive answer
    for the given question using only the information available in the context.

    context: {context}

    question: {question}

    <response> Your answer in Markdown format. </response>
    """
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain_type_kwargs = {"prompt": prompt}

                # Initialize the appropriate LLM based on selection
                if llm_choice == "Closed Source":
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, max_tokens=1000)
                else:
                    llm = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.3)

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs=chain_type_kwargs,
                    verbose=True,
                )

                # Chat interface using streamlit_chat
                user_input = st.text_input("Your Message:", key="chat_input")
                if st.button("Send", key="send_button") and user_input:
                    response = qa(user_input)
                    bot_answer = response["result"]
                    st.session_state.chat_history.append({"user": user_input, "bot": bot_answer})

                    # Save the chat history to a file (chat_history.txt) for this session
                    chat_file_content = "\n\n".join(
                        [f"User: {chat['user']}\nBot: {chat['bot']}" for chat in st.session_state.chat_history]
                    )
                    with open("chat_history.txt", "w", encoding="utf-8") as cf:
                        cf.write(chat_file_content)

                # Display the conversation using streamlit_chat component
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st_message(chat["user"], is_user=True)
                        st_message(chat["bot"], is_user=False)
            else:
                st.info("Please create embeddings to activate the chat.")

elif page == "Contact":
    st.markdown("""
    ## Contact

    - **GitHub:** [AIAnytime](https://github.com/AIAnytime)  
      _Contributions are welcome – fork the repository and submit a PR!_

    - **LinkedIn:** [Sonu Kymar](https://www.linkedin.com/in/sonukr0/)  
      _Feel free to reach out on LinkedIn for collaborations or questions._

    - **YouTube:** Watch the demonstration video below!
    """)
    st.video("https://youtu.be/5wJsW9LGgr8")
