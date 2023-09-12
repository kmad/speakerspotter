#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
from dotenv import load_dotenv
from pydub import AudioSegment
import subprocess
import replicate
import argparse
import logging
import json
import uuid
import os

def generate_embeddings(run_id, speakers):
    logging.info("Embedding sentences in transcripts...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_path = os.getenv("embeddings_file")

    # TODO: Replace with a more scalable solution
    output_object  = {
            "run_id": run_id,
            "speakers": list(speakers),
            "data" : {}
        }

    for speaker_label in speakers:
        transcript_file = f"{run_id}/{speaker_label}-TRANSCRIPT.txt"

        with open(transcript_file, 'r') as f:
            text = f.read().replace('\n', ' ')
            sentences = [sentence.strip() for sentence in text.split('. ') if sentence]

        # TODO: Offload to GPU
        embeddings = model.encode(sentences)
        
        logging.info(f"Generated embeddings: {len(embeddings)}")
        
        # TODO: replace with np.save or a vector db, when appropriate
        output_object["data"][speaker_label] = [x for x in embeddings.tolist()]
        
    # TODO: Use a more mature solution
    with open(embeddings_path, 'a') as f:
        f.write(json.dumps(output_object))
        f.write('\n')
        
    logging.info("All sentences in transcripts embedded.")

def process_speaker(run_id, speaker_label):
    audio_input_path = f'{run_id}/{speaker_label}.wav'
    audio_output_path = f'{run_id}/{speaker_label}_16k.wav'
    transcript_output_path = f'{run_id}/{speaker_label}-TRANSCRIPT' # Whisper will append .txt

    WHISPER_PATH = os.getenv("WHISPER_PATH")
    MODEL_PATH   = os.getenv("MODEL_PATH")

    # Resample the audio to 16kHz, which is required by whisper
    subprocess.run(["ffmpeg"
                    , "-i", audio_input_path
                    , "-acodec", "pcm_s16le"
                    , "-ac", "1"
                    , "-ar", "16000"
                    , "-threads", "20"
                    , audio_output_path]
                    , stdout=subprocess.STDOUT if args.verbose else subprocess.DEVNULL
                    , stderr=subprocess.STDOUT if args.verbose else subprocess.DEVNULL)
    
    logging.info(f"Converted audio for speaker {speaker_label} saved to {audio_output_path}")
     
    # Delete the original file
    subprocess.run(["rm", audio_input_path])

    logging.info("Running whisper...")

    subprocess.run([WHISPER_PATH
                    , "-m"
                    , MODEL_PATH
                    , "-f", f"{audio_output_path}"
                    , "-otxt"
                    , "-of", f"{transcript_output_path}"]
                    , stdout=subprocess.STDOUT if args.verbose else subprocess.DEVNULL
                    , stderr=subprocess.STDOUT if args.verbose else subprocess.DEVNULL
                    )
    
def extract_all_samples_per_speaker(run_id, audio_path):

    logging.info("Extracting all samples per speaker...")
    speakers_segments = {}

    rttm_path = f"{run_id}/output.rttm"
    with open(rttm_path, 'r') as rttm_file:
        for line in rttm_file:
            parts = line.split()
            if len(parts) >= 5:
                speaker_label = parts[7]
                start_time = float(parts[3]) * 1000
                duration = float(parts[4]) * 1000
                end_time = start_time + duration
                if speaker_label not in speakers_segments:
                    speakers_segments[speaker_label] = [(start_time, start_time + duration)]
                else:
                    speakers_segments[speaker_label].append((start_time, start_time + duration))
    
    #TODO: Support other formats (convert with ffmpeg?)
    logging.info("Loading audio file...")
    original_audio = AudioSegment.from_wav(audio_path)
    
    logging.info("Segmenting audio...")
    speakers_audio = {}
    for speaker_label, segments in speakers_segments.items():
        for start_time, end_time in segments:
            segment_audio = original_audio[start_time:end_time]
            if speaker_label not in speakers_audio:
                speakers_audio[speaker_label] = segment_audio
            else:
                speakers_audio[speaker_label] += segment_audio

    for speaker_label, speaker_audio in speakers_audio.items():
        output_path = f'{run_id}/{speaker_label}.wav'
        logging.info(f"Exporting segment for speaker {speaker_label} to {output_path}")
        speaker_audio.set_channels(1)
        speaker_audio.set_frame_rate(16000)
        speaker_audio.export(output_path, format="wav")


    with Pool() as p:
        p.starmap(process_speaker, [(run_id, speaker_label) for speaker_label in speakers_audio.keys()])
        
    logging.info("All audio segments processed.")

    logging.info("Generating embeddings...")
    generate_embeddings(run_id, speakers_audio.keys())
    
def split_audio(run_id, audio_path):
    output = replicate.run(
    "lucataco/speaker-diarization:718182bfdc7c91943c69ed0ac18ebe99a76fdde67ccd01fced347d8c3b8c15a6",
    input={"audio": open(audio_path, "rb")}
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    load_dotenv()
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the necessary command-line arguments
    parser.add_argument('-i', '--input', help='Path to the audio file')
    parser.add_argument('-v', '--verbose', help='Turn on verbose output of ffmpeg and whisper', action='store_true')

    global args

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set the input parameters
    audio_path = args.input
    # Add error checks for the cli input
    if not os.path.exists(audio_path):
        logging.error("Audio file not found: ", audio_path)
        exit(1)
    
    # Create a unique identifier for this run
    run_id = str(uuid.uuid4()) 

    # Create a directory for this run
    os.mkdir(run_id)
    
    logging.info(f"Using Run ID: {run_id}")

    # Use pyannote to extract one sample per speaker
    logging.info("Starting audio processing...")

    split_audio(run_id, audio_path)

    extract_all_samples_per_speaker(run_id, audio_path)

    # TODO: Youtube download / extract / convert to wav
