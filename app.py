import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv() 

## Langsmith Tracking
os.environ["LANGSMITH_API_KEY"]=st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING_V2"]="true"
os.environ["LANGSMITH_PROJECT"]="Youtube Video Summarizer"

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("Gen-AI: Summarize Text From YouTube or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    google_api_key=st.text_input("Google API Key",value=st.secrets["GOOGLE_API_KEY"],type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model Using Google API
llm =ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", google_api_key=google_api_key)

prompt_template="""
You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if "www.youtube.com" in generic_url:
    video_id = generic_url.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not google_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video url or website url")
    else:
        try:
            with st.spinner("Loading..."):
                ## loading the website or yt video data
                if "www.youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=True,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                
                output_summary=chain.invoke(docs)

                st.write(output_summary['output_text'])
        except Exception as e:
            st.exception(f"Exception:{e}")
