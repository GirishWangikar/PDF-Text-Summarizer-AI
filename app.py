import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import PyPDF2
from langchain.prompts import PromptTemplate

# Set up API keys
groq_api_key = os.environ.get('GROQ_API_KEY')

# Set up LLM
llm = ChatGroq(temperature=0, model_name='llama-3.1-8b-instant', groq_api_key=groq_api_key)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def summarize_chunks(chunks, conciseness):
    # Adjust the prompts based on the conciseness level
    map_prompt_template = f"""Write a {'very concise' if conciseness > 0.7 else 'detailed'} summary of the following text, focusing on the {'most crucial' if conciseness > 0.7 else 'key'} points:
    "{{text}}"
    {'CONCISE' if conciseness > 0.7 else 'DETAILED'} SUMMARY:"""

    combine_prompt_template = f"""Write a {'highly condensed' if conciseness > 0.7 else 'comprehensive'} summary of the following text, capturing the {'essential' if conciseness > 0.7 else 'key'} points and main ideas:
    "{{text}}"
    {'CONDENSED' if conciseness > 0.7 else 'COMPREHENSIVE'} SUMMARY:"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    # Adjust the chain type based on the document length and conciseness
    total_length = sum(len(chunk.page_content) for chunk in chunks)
    if total_length < 10000 or conciseness > 0.8:
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=combine_prompt
        )
    else:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )

    summary = chain.run(chunks)
    return summary

def summarize_content(pdf_file, text_input, conciseness):
    if pdf_file is None and not text_input:
        return "Please upload a PDF file or enter text to summarize."

    if pdf_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
    else:
        # Use the input text
        text = text_input

    # Chunk the text
    chunks = chunk_text(text)

    # Summarize chunks with conciseness level
    final_summary = summarize_chunks(chunks, conciseness)
    return final_summary

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        <h1 style="text-align: center;">PDF And Text Summarizer</h1>
        <h3 style="text-align: center;">Advanced PDF and Text Summarization with Conciseness Control - Upload your PDF document or enter text directly, adjust the conciseness level, and let AI generate a summary.</h3>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_pdf = gr.File(label="Upload PDF (optional)", file_types=[".pdf"])
            input_text = gr.Textbox(label="Or enter text here", lines=5, placeholder="Paste or type your text here...")
            conciseness_slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Conciseness Level")
            submit_btn = gr.Button("Generate Summary", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="Generated Summary", lines=10)

    gr.Markdown(
        """
        ### How it works
        1. Upload a PDF file or enter text directly
        2. Adjust the conciseness level:
           - 0 (Most detailed) to 1 (Most concise)
        3. Click "Generate Summary"
        4. Wait for the AI to process and summarize your content
        5. Review the generated summary
        *Powered by LLAMA 3.1 8B model and LangChain*
        """
    )

    gr.HTML(
        """
        <footer> 
            <p>If you enjoyed the functionality of the app, please leave a like!<br>
            Check out more on <a href="https://www.linkedin.com/in/girish-wangikar/" target="_blank">LinkedIn</a> | 
            <a href="https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/" target="_blank">Portfolio</a></p>
        </footer>
        """
    )

    submit_btn.click(summarize_content, inputs=[input_pdf, input_text, conciseness_slider], outputs=output)

iface.launch()
