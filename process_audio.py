#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import shutil
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pydub import AudioSegment
import subprocess
import replicate
import argparse
import requests
import logging
import random
import openai
import json
import uuid
import os

import numpy as np
import faiss

class DefaultList(list):
    def __init__(self, default_value, *args):
        self.default_value = default_value
        super().__init__(*args)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except IndexError:
            return self.default_value
        
def insert_embeddings_to_faiss_db(embeddings, run_id, speaker_list):
    dimension = len(next(iter(embeddings.values())))  # get the dimension size from the first embedding vector
    
    # Load FAISS database
    if os.getenv("faiss_db_file_path") is not None:
        db_file_path = os.getenv("SPEAKER_DB_PATH")
    else:
        db_file_path = f"speaker_db.index"  # define the path to save the faiss database

    if os.path.exists(db_file_path):  # if the database file exists
        db = faiss.read_index(db_file_path)  # load the database from the file
    else:
        db = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))  # create a new faiss database with L2 distance metric

    # Load speaker map
    try:
        with open('speaker_map.json', 'r') as f: # TODO: Make configurable
            speaker_map = json.load(f)
    except:
        logging.warn("First run? Speaker map not found. Creating new speaker map.")
        speaker_map = {}

    count = 0
    for speaker, embedding in embeddings.items():
        # Create a unique ID for each speaker using the run_id and speaker label
        speaker_id = int(hashlib.sha1(f"{run_id}_{speaker}".encode()).hexdigest(), 16) % (10 ** 8)
        logging.info(f"Speaker {speaker} has ID {speaker_id} // {run_id}")

        speaker_label = f"Unknown-{speaker_id}"
        if len(speaker_list) > 0:
            speaker_label = speaker_list[count]
            logging.info(f"Processing labeled speaker {speaker_label}")
            

        embedding_np = np.asarray(embedding).astype('float32')  # convert the embedding to numpy array
        embedding_np = np.expand_dims(embedding_np, axis=0)  # add an extra dimension for faiss


        if db.ntotal > 0:  # if the database is not empty
            
            # compute the cosine distance between the new embedding and all existing embeddings in the database
            distances, _ = db.search(embedding_np, db.ntotal) # TODO: Do we need all of these?
            if np.min(distances) < 0.01:  # if the minimum distance is less than 0.01, the exact embedding exists
                logging.info(f"Speaker {speaker} already exists in the database.")
                
            else:
                logging.info(f"Speaker {speaker} embedding is new.")
            logging.info(f"{speaker} min distance: {np.min(distances)}")


            #########
            # Search for similar embeddings
            THRESHOLD = 45_000
            r = db.range_search(embedding_np, THRESHOLD)
            num_results = r[0][1]
            id_list = list(r[2])
            
            # Get the speaker name from the speaker map
            identified_speakers = set()
            for _id in id_list:
                to_add = speaker_map.get(str(_id), "Unknown")
                if "Unknown" not in to_add:
                    identified_speakers.add(to_add)

            if len(identified_speakers) > 0 and "Unknown" in speaker_label:
                logging.info(f"Unknown speaker found: using {list(identified_speakers)[0]}.")
                speaker_label = list(identified_speakers)[0]
            elif len(identified_speakers) > 0 and list(identified_speakers)[0] == speaker_label:
                logging.info("Speaker DB agrees with provided label.")
            elif len(identified_speakers) > 0 and list(identified_speakers)[0] != speaker_label:
                logging.warning(f"Speaker DB disagrees with provided label. Provided: {speaker_label} // Found:{list(identified_speakers)}.")
            elif len(identified_speakers) > 0:
                logging.info(f"Identified speakers: {identified_speakers} across {num_results} entries.")
            
        
        # Add the speaker to the speaker map
        speaker_map[str(speaker_id)] = speaker_label
        db.add_with_ids(embedding_np, np.array([speaker_id]))  # add the embedding to the database with the unique speaker_id

        # Update the speaker map
        speaker_names = set([speaker_map[str(i)] for i in id_list])
        # Filter out anything starting with "Unknown"
        speaker_names = [name for name in speaker_names if not name.startswith("Unknown")]
        # If we have one result, that's the name to replace all other unknowns with
        if len(speaker_names) == 1:
            # Update instances of the speaker map from id_list, using the name
            speaker_map = {**speaker_map, **{str(i): speaker_names[0] for i in id_list}}
            logging.info(f"Updated {len(id_list)} entries with speaker name {speaker_names[0]}")
        elif len(speaker_names) > 1:
            print(f"Found multiple speakers: {speaker_names}. This requires a deconflict.")
            
        count += 1

    faiss.write_index(db, db_file_path)  # save the database to a file
    
    # Write the speaker_map back to file
    with open('speaker_map.json', 'w') as f:
        json.dump(speaker_map, f)

    return db

def process_segments_json(run_id, speaker_list):
    with open(f'{run_id}/segments.json', 'r') as f:
        data = json.load(f)

    embeddings = data['speakers']['embeddings']
    db = insert_embeddings_to_faiss_db(embeddings, run_id, speaker_list)

    return db


def generate_embeddings(run_id, speakers_text):
    logging.info("Embedding sentences in transcripts...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_path = os.getenv("embeddings_file")

    # TODO: Replace with a more scalable solution
    output_object  = {
            "run_id": run_id,
            "speakers": list(speakers_text.keys()),
            "data" : {}
        }

    for speaker_label in speakers_text.keys():

        sentences = [sentence.strip() for sentence in speakers_text[speaker_label].split('. ') if sentence]

        # TODO: Offload to GPU or AI call
        embeddings = model.encode(sentences)
        
        logging.info(f"Generated embeddings: {len(embeddings)}")
        
        # TODO: replace with np.save or a vector db, when appropriate
        output_object["data"][speaker_label] = [x for x in embeddings.tolist()]
        
    # TODO: Use a more mature solution
    with open(embeddings_path, 'a') as f:
        f.write(json.dumps(output_object))
        f.write('\n')
        
    logging.info("All sentences in transcripts embedded.")
    
def extract_all_samples_per_speaker(run_id):
    INPUT_PATH = f'{run_id}/input.mp3' 

    logging.info("Extracting all samples per speaker...")
    speakers_segments = {}


    with open(f'{run_id}/segments.json', 'r') as f:
        segment_output = json.load(f)

    data = segment_output

    # Iterate over the segments
    for segment in data['segments']:
        speaker = segment['speaker']
        start_time = segment['start']
        stop_time = segment['stop']

        # Convert timestamps to milliseconds
        start_time_parts = start_time.split(':')
        start_time_ms = int(start_time_parts[0]) * 60 * 60 * 1000 + int(start_time_parts[1]) * 60 * 1000 + int(float(start_time_parts[2]) * 1000)

        stop_time_parts = stop_time.split(':')
        stop_time_ms = int(stop_time_parts[0]) * 60 * 60 * 1000 + int(stop_time_parts[1]) * 60 * 1000 + int(float(stop_time_parts[2]) * 1000)

        # Add the segment to the speaker's list
        if speaker not in speakers_segments:
            speakers_segments[speaker] = []
        speakers_segments[speaker].append((start_time_ms, stop_time_ms))
    

    logging.info("Loading audio file...")
    original_audio = AudioSegment.from_mp3(INPUT_PATH)
    
    logging.info("Segmenting audio...")
    speakers_audio = {}
    speakers_text = {}

    for speaker_label, segments in speakers_segments.items():
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Define a lock object to synchronize threads
        lock = threading.Lock()

        # Define a function to process each segment
        def process_segment(segment):
            start_time, end_time = segment
            segment_audio = original_audio[start_time:end_time]

            # Generate transcript for segment_audio
            # Use openai model
            # Generate a random number or other identifier to allow for running in parallel
           
            temp_filename = f"/tmp/temp_audio_{random.randint(1, 100000)}.mp3"
            
            segment_audio.export(temp_filename, format="mp3")
            duration_seconds = segment_audio.duration_seconds
            # Skip if temp_filename is more than 20MB
            # TODO: don't skip; should be able to handle this with mp3s
            logging.info(f"Processing segment of {duration_seconds} seconds.")
            if os.path.getsize(temp_filename) > 20000000:
                print(f"Skipping segment {start_time}-{end_time} for speaker {speaker_label} because it is too large.")
                return

            try:
                with open(temp_filename, 'rb') as audio_file:
                    output = openai.Audio.transcribe("whisper-1", audio_file)
            except:
                print(f"Split segment {start_time}-{end_time} for speaker {speaker_label} because it failed to transcribe.")
            
                # Get the full length of the segment
                segment_length = end_time - start_time
                
                from io import BytesIO
                # Split the segment in half and attempt to transcribe each half
                segment_audio_1 = segment_audio[:segment_length//2]
                # Transcribe the first half of the segment
                audio_file_1 = BytesIO()
                segment_audio_1.export(audio_file_1, format="mp3")
                audio_file_1.seek(0)
                transcription_1 = openai.Audio.transcribe("whisper-1", audio_file_1)

                # Get the second half of the segment
                segment_audio_2 = segment_audio[segment_length//2:]

                # Transcribe the second half of the segment
                audio_file_2 = BytesIO()
                segment_audio_2.export(audio_file_2, format="mp3")
                audio_file_2.seek(0)
                transcription_2 = openai.Audio.transcribe("whisper-1", audio_file_2)

                # Combine the transcriptions
                full_transcription = transcription_1 + " " + transcription_2
                output = {"text": full_transcription}
                
            # Add to speaker's transcript
            with lock:
                if speaker_label not in speakers_text:
                    speakers_text[speaker_label] = [(start_time, output['text'] + " ")]
                else:
                    speakers_text[speaker_label].append((start_time, output['text'] + " "))

                # Add audio segments
                if speaker_label not in speakers_audio:
                    speakers_audio[speaker_label] = [(start_time, segment_audio)]
                else:
                    speakers_audio[speaker_label].append((start_time, segment_audio))

        # Use a ThreadPoolExecutor to run the process_segment function in parallel for each segment
        with ThreadPoolExecutor() as executor:
            executor.map(process_segment, segments)


     # Sort the audio segments by start_time to ensure they are in order
    
    for speaker_label in speakers_audio.keys():
        speakers_audio[speaker_label].sort(key=lambda x: x[0])
        speakers_audio[speaker_label] = sum([audio for _, audio in speakers_audio[speaker_label]])
    
    for speaker_label, speaker_audio in speakers_audio.items():
            # Sort the transcripts by start_time to ensure they are in order
        speakers_text[speaker_label].sort(key=lambda x: x[0])
        speakers_text[speaker_label] = " ".join([text for _, text in speakers_text[speaker_label]])
        
        # Output transcript
        transcript_path = f'{run_id}/{speaker_label}-TRANSCRIPT.txt'
        logging.info(f"Exporting transcript for speaker {speaker_label} to {transcript_path}")

        with open(transcript_path, "w") as f:
            f.write(speakers_text[speaker_label])

        # Output audio
        output_path = f'{run_id}/{speaker_label}.mp3'
        logging.info(f"Exporting segment for speaker {speaker_label} to {output_path}")
        speaker_audio.export(output_path, format="mp3")
        
    logging.info("All audio segments processed.")

    logging.info("Generating embeddings...")

    generate_embeddings(run_id, speakers_text)
    
def diarize_audio(run_id, audio_path):
 
    logging.info("Starting diarization...")

    # Use replicate as the hosted diarization solution
    output_location = replicate.run(
        "lucataco/speaker-diarization:718182bfdc7c91943c69ed0ac18ebe99a76fdde67ccd01fced347d8c3b8c15a6",
        input={"audio": audio_path}
        #input={"audio": open(audio_path, "rb")}
    )
    
    logging.info("Diarization complete.")

    # Replicate saves the results elsewhere
    r = requests.get(output_location)
    # Save output to run_id folder
    with open(f'{run_id}/segments.json', "w") as f:
        json.dump(r.json(), f)
    
    return output_location

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the necessary command-line arguments
    parser.add_argument('-i', '--input', help='Path to the audio file')
    parser.add_argument('-v', '--verbose', help='Turn on verbose output of ffmpeg and whisper', action='store_true')
    parser.add_argument('-s', '--speakers', help='Comma separated list of speakers in order of appearance', type=str)

    global args

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set the input parameters
    audio_path = args.input
    
    try:
        speaker_list = DefaultList('Unknown', args.speakers.split(','))
    except:
        speaker_list = DefaultList('Unknown', [])
        
    # Create a unique identifier for this run
    
    run_id = str(uuid.uuid4()) 

    # Create a directory for this run
    os.mkdir(run_id)
    
    logging.info(f"Using Run ID: {run_id}")

    # Use pyannote to extract one sample per speaker
    logging.info("Starting audio processing...")
    

    logging.info("Downloading audio file from URL...")
    
    if "youtube.com" in audio_path:
        logging.info("Downloading from YouTube...")
        ENDPOINT = os.getenv("YT_ENDPOINT")
        full_url = ENDPOINT + audio_path
        audio_path = requests.get(full_url).text
        logging.info(f"Using generated audio URL {audio_path}")

    r = requests.get(audio_path)
    with open(f"{run_id}/input.mp3", 'wb') as f:
        f.write(r.content)
    
    diarize_audio(run_id, audio_path)

    extract_all_samples_per_speaker(run_id)

    process_segments_json(run_id, speaker_list)

