# PDF and Text Summarizer AI

This project is an advanced PDF and text summarization tool with conciseness control. It uses the LLAMA 3.1 8B model and LangChain to generate summaries from PDF documents or text input.

## Features

- Summarize PDF documents or text input
- Adjust conciseness level of the summary
- User-friendly Gradio interface
- Powered by LLAMA 3.1 8B model and LangChain

## Installation

1. Clone this repository:git clone https://github.com/yourusername/pdf-text-summarizer.git

cd pdf-text-summarizer

2. Install the required packages:
pip install -r requirements.txt

3. Set up your Groq API key as an environment variable:
export API_KEY='your-api-key-here'

## Usage
Run the application:
python app.py

## How it works
1. Upload a PDF file or enter text directly
2. Adjust the conciseness level (0 for most detailed, 1 for most concise)
3. Click "Generate Summary"
4. Wait for the AI to process and summarize your content
5. Review the generated summary

## Dependencies
- os
- gradio
- langchain_groq
- langchain
- PyPDF2
