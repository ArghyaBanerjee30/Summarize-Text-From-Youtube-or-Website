import streamlit as st
import validators
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


# Title
st.set_page_config(page_icon="🦜", page_title="LangChain: Summarize Text From Youtube or Website")
st.title("🦜LangChain: Summarize Text From Youtube or Website")
st.subheader("Summarize URL")

# Inputs
groq_api_key = st.sidebar.text_input("Groq API key:", type="password")
url = st.text_input("URL", label_visibility="collapsed")

# Define LLM Model
if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    template = """
        Provide a summary of the following content in 300 words: 
        Content: {text}
    """

    # Create the prompt template object
    stuff_prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )

    if st.button("Summarise"):
        # Validate all the inputs
        if not url.strip():
            st.error("Please provide the URL")
        elif not validators.url(url):
            st.error("Please enter a valid URL")
        else:
            try:
                with st.spinner("Processing..."):
                    # Determine the loader based on the URL type
                    if "youtube.com" in url:
                        loader = YoutubeLoader.from_youtube_url(
                            url, 
                            add_video_info = True
                        )
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[url], 
                            ssl_verify = False, 
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                            }
                        )
                    
                    # Load the content from the URL
                    data = loader.load()

                    # Load the summarize chain using the 'stuff' method
                    stuff_llm_chain = load_summarize_chain(
                        llm=llm, 
                        chain_type='stuff', 
                        prompt=stuff_prompt, 
                    )
                    stuff_summary = stuff_llm_chain.run(data)

                    st.success(stuff_summary)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
else:
    st.error("Please provide the API Key")