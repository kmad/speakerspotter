# SpeakerSpotter

SpeakerSpotter is a Python application that uses machine learning to transcribe and diarize audio files. It is particularly useful for processing large audio files with multiple speakers (typically podcasts, interviews, etc.).

## Features

- Transcribes audio files using OpenAI's whisper.
- Diarizes (identifies and separates speakers) in the audio file using PyAnnote.
- Splits the input audio into segments by speaker using `pydub`
- Supports parallel processing for faster transcription and diarization.
- Stores embeddings for speaker fingerprints (faiss) to detect speakers by voice, embeddings of transcripts (chroma), and transcript segments (sqlite).

## Dependencies

- A Replicate API key
- `python3`
- OpenAI's Whisper (optional, if running locally)
- `SQLite`
- `FAISS` for efficient similarity search of speaker embeddings
- `pydub` for audio processing
- `chromadb` for persistent storage of text embeddings

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Move `.env.example` to `.env` and populate the necessary environment variables (REPLCIATE_API_KEY, DB_PATH, CHROMA_PATH, etc.).
  - An optional `YT_ENDPOINT` path specifies a service that takes as input a Youtube URL and returns a link to an audio file, which is then processed. This is only used if the URL starts with `youtube.com`.
4. Run the script `process_audio.py`. For example:
`python3 process_audio.py -ll DEBUG -s "Speaker1,Speaker2" -i http://PATH_TO_YOUR_AUDIO_FILE`
  - The program will automatically match existing speakers to labels if they have been processed at least once before, and do not require a label.
  - For unknown speakers you can specify using `-s "Speaker1,Unknown,Speaker3`
  - Any speakers *after* the last provided speaker label will default to Unknown and will result in a lookup. E.g., if there are three speakers, using `-s Speaker1` will label only the first speaker, and the rest will be detected or default to Unknown.
5. You can then check the result in the sqlite3 database:
`SELECT * FROM segments WHERE run_id = '<RUN_ID'> ORDER BY start_time asc`

## Things to note/ TODO
- This is alpha software and is absolute garbage in certain places. Not ready for production at all. Feel free to improve - it needs it.
- There's no need to have both Chroma and FAISS (and arguably sqlite), these should be consolidated into a single DB
- Needs better logic to resolve mis-matched speaker labels (e.g., what's in the db vs. what's provided)
- Needs a better interface, either a web interface (in the works), output to csv, etc.