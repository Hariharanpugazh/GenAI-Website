from flask import Flask, request, render_template, send_file
from gtts import gTTS
import speech_recognition as sr
import requests
import google.generativeai as genai
from pathlib import Path
from pydub import AudioSegment
import os
app = Flask(__name__)

# Set up the Google Generative AI and Gemini Vision Pro configurations
genai.configure(api_key="AIzaSyDxHEMIcdmMLa7Zd-uiGNffPDP4mUo8bR4")

generation_config = {
    "temperature": 0.05,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1000,
}

# Safety settings for Gemini AI
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
]

# Set up Gemini Vision Pro API
GEMINI_VISION_PRO_API_KEY = 'AIzaSyDxHEMIcdmMLa7Zd-uiGNffPDP4mUo8bR4'
GEMINI_VISION_PRO_ENDPOINT = 'https://api.geminivisionpro.com/v1/process'

# Folder to save uploads
UPLOAD_FOLDER = Path("uploads/")
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Handle Text Input to AI Response
@app.route('/generate-text', methods=['POST'])
def generate_text():
    user_input = request.form.get('user_text')

    # Using Google Generative AI to generate a response
    model = genai.GenerativeModel(model_name="gemini-pro", 
                                  generation_config=generation_config, 
                                  safety_settings=safety_settings)
    response = model.generate_content([user_input])

    return render_template('index.html', ai_response=response.text)

# Handle Text-to-Speech and Speech-to-Text
@app.route('/speech', methods=['POST'])
def handle_speech():
    if 'voice_file' in request.files:
        voice_file = request.files['voice_file']
        file_path = UPLOAD_FOLDER / 'voice_message.ogg'
        voice_file.save(file_path)

        # Convert speech to text
        recognizer = sr.Recognizer()
        audio = AudioSegment.from_file(file_path)
        audio.export("voice_message.wav", format="wav")
        with sr.AudioFile("voice_message.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Convert the text back to speech
        tts = gTTS(text=text, lang='en')
        tts.save("text_to_speech.mp3")

        return send_file("text_to_speech.mp3", as_attachment=True)

    return "No voice file provided", 400

# Handle Image Upload and Processing
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image_file' in request.files:
        image_file = request.files['image_file']
        image_path = UPLOAD_FOLDER / image_file.filename
        image_file.save(image_path)

        # Send the image to the Gemini Vision Pro API for processing
        with open(image_path) as img:
            files = {'image': img}
            headers = {'x-api-key': GEMINI_VISION_PRO_API_KEY}
            response = requests.post(GEMINI_VISION_PRO_ENDPOINT, files=files, headers=headers)

        result = response.text
        st.write(result)
        return render_template('index.html', image_result=result)

    return "No image file provided", 400

if __name__ == '__main__':
    app.run(debug=True)
