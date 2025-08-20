# GenAI Tools - Gemini API Integration

A Flask web application that provides AI-powered features using Google's Gemini API, including text generation, image processing, and speech-to-text capabilities.

## Features

- **Text Generation**: Generate AI responses using Google's Gemini Pro model
- **Image Processing**: Process and analyze images using Gemini Vision Pro API
- **Speech Processing**: Convert speech to text and text back to speech
- **Web Interface**: Clean, responsive web interface for all features

## Technologies Used

- **Backend**: Flask (Python)
- **AI Integration**: Google Generative AI (Gemini Pro, Gemini Vision Pro)
- **Speech Processing**: Google Text-to-Speech (gTTS), SpeechRecognition
- **Audio Processing**: pydub
- **Frontend**: HTML, CSS
- **File Handling**: Pathlib, os

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hariharanpugazh/GenAI-Tools-Gemini-API-.git
cd GenAI-Tools-Gemini-API-
```

2. Install required dependencies:
```bash
pip install flask gtts speechrecognition requests google-generativeai pydub
```

3. Configure API keys:
   - Obtain a Google Generative AI API key
   - Update the API key in `app.py`:
     ```python
     genai.configure(api_key="your_google_api_key_here")
     GEMINI_VISION_PRO_API_KEY = 'your_gemini_vision_pro_api_key_here'
     ```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the available features:
   - **Generate Text**: Enter a text prompt to get AI-generated responses
   - **Process Image**: Upload an image for AI analysis
   - **Speech Processing**: Upload audio files for speech-to-text conversion

## Project Structure

```
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Web interface template
├── static/
│   └── style.css       # Styling for the web interface
├── uploads/            # Directory for uploaded files
└── README.md           # Project documentation
```

## API Endpoints

- `GET /` - Main page with web interface
- `POST /generate-text` - Generate text using Gemini Pro
- `POST /process-image` - Process images using Gemini Vision Pro
- `POST /speech` - Handle speech-to-text and text-to-speech conversion

## Configuration

The application includes safety settings for content filtering and generation parameters that can be adjusted in `app.py`:

- Temperature: 0.05 (controls randomness)
- Max output tokens: 1000
- Safety thresholds for harmful content categories

## Requirements

- Python 3.7+
- Google Generative AI API access
- Internet connection for API calls

## Notes

- Ensure you have valid API keys before running the application
- The application creates an `uploads/` directory for temporary file storage
- Audio files are converted to WAV format for speech recognition processing

## License

This project is open source and available under the MIT License.