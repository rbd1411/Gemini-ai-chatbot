from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PIL.Image
import os
import re

# Load both text and vision models
txt_model = genai.GenerativeModel('gemini-pro')
vis_model = genai.GenerativeModel('gemini-pro-vision')

os.environ['GOOGLE_API_KEY'] = 'your api key'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize SQLite database
DATABASE = 'chatbot_data.db'

def create_database():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT,
        response TEXT,
        image BLOB
    )''')
    conn.commit()
    conn.close()

create_database()

def add_chat_history(prompt, response, image):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''INSERT INTO chat_history (prompt, response, image) VALUES (?, ?, ?)''',
              (prompt, response, image))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''SELECT * FROM chat_history''')
    rows = c.fetchall()
    conn.close()
    return rows

# Routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/chat')
def chat():
    return render_template('index.html', chat_history=get_chat_history())

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('home'))

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            prompt = request.form['prompt']
            image = request.files.get('image')  # Get the uploaded image
            pdf_file = request.files.get('pdf_file')  # Get the uploaded PDF file

            if pdf_file:
                # Process PDF file
                pdf_text = get_pdf_text(pdf_file)
                text_chunks = get_text_chunks(pdf_text)
                response = txt_model.generate_content(f"{prompt}\n\nContext:\n{' '.join(text_chunks)}")
            elif image:
                # Use Gemini Pro Vision for image-based prompts
                img = PIL.Image.open(image)
                response = vis_model.generate_content([prompt, img])
            else:
                # Use Gemini Pro for text-only prompts
                response = txt_model.generate_content(prompt)

            # Check if the prompt is a list or code
            is_list = re.search(r'^\s*\d+\.', prompt, re.MULTILINE)
            is_code = re.search(r'```', prompt)

            if is_list:
                # Format response as a list
                formatted_response = '\n'.join([f'{i+1}. {line}' for i, line in enumerate(response.text.split('\n'))])
            elif is_code:
                # Format response as code
                formatted_response = f'```\n{response.text}\n```'
            else:
                # No special formatting
                formatted_response = response.text

            if formatted_response:
                # Add chat history to the database
                add_chat_history(prompt, formatted_response, image.read() if image else None)
                return formatted_response
            else:
                return "Sorry, Gemini didn't have a response!"
        except Exception as e:
            return "Sorry, an error occurred!"

    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)
