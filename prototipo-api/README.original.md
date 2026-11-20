# Whisper Transcriber & Diarizer

A complete solution for transcribing audio files and identifying speakers using OpenAI's Whisper and speaker diarization technology.

## Features

- **Audio Transcription**: Convert speech to text using OpenAI's Whisper models
- **Speaker Diarization**: Identify and label different speakers in audio recordings
- **Easy-to-Use Interface**: Simple GUI for uploading and processing audio files
- **Multiple Languages**: Support for various languages
- **Model Selection**: Choose from different Whisper model sizes based on accuracy needs
- **Flexible Deployment**: Run via Docker containers or directly as Python applications

## Project Structure

```
whisper_transcriber_diarizer/
├── backend/               # FastAPI backend service
│   ├── app/               # Application code
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core configurations
│   │   └── services/      # Business logic
│   ├── main.py            # Entry point
│   ├── requirements.txt   # Dependencies
│   └── tests/             # Unit tests
├── frontend/              # Tkinter GUI
│   ├── src/               # Source code
│   │   ├── components/    # UI components
│   │   ├── utils/         # Utilities
│   │   └── gui.py         # Main application
│   └── requirements.txt   # Dependencies
├── docker/                # Docker configuration
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
└── README.md
```

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/PoltorProgrammer/Whisper_Transcriber_Diarizer.git
   cd Whisper_Transcriber_Diarizer
   ```

2. Start the services:
   ```bash
   cd docker
   docker-compose up -d
   ```

3. Access the API at http://localhost:8000/api/v1

### Running Locally

#### Backend

1. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```bash
   python main.py
   ```

#### Frontend

1. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

2. Run the GUI:
   ```bash
   python src/gui.py
   ```

## Development

### Testing Version

For development and testing, the application uses a lightweight version without downloading large models. This allows for faster testing of the application flow.

To use the testing version:
- Ensure you're using the mock `transcribe_audio` function in `backend/app/services/transcription.py`
- The Docker configuration comments out the Whisper installation

### Production Version

For the full-featured version:
- Uncomment the Whisper installation in the Docker configuration
- Replace the mock transcription function with the actual implementation
- Install additional dependencies for diarization

## Required Dependencies

### Backend
- FastAPI
- Uvicorn
- Python-multipart
- OpenAI Whisper (for production)
- PyAnnote Audio (for diarization)
- FFmpeg

### Frontend
- Tkinter
- Requests

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
