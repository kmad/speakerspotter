#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pydub import AudioSegment
import replicate
import argparse
import requests
import logging
import openai
import json
import uuid
import os

import numpy as np
import faiss

def insert_embeddings_to_faiss_db(embeddings, run_id, speaker_list):
    dimension = len(next(iter(embeddings.values())))  # get the dimension size from the first embedding vector
    
    # Load FAISS database
    db_file_path = f"faiss_db.index"  # define the path to save the faiss database
    if os.path.exists(db_file_path):  # if the database file exists
        db = faiss.read_index(db_file_path)  # load the database from the file
    else:
        db = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))  # create a new faiss database with L2 distance metric

    # Load speaker map
    try:
        with open('speaker_map.json', 'r') as f: # TODO: Make configurable
            speaker_map = json.load(f)
    except:
        speaker_map = {}

    count = 0
    for speaker, embedding in embeddings.items():
        if len(speaker_list) > 0:
            logging.info(f"Processing labeled speaker {speaker_list[count]}")

        embedding_np = np.asarray(embedding).astype('float32')  # convert the embedding to numpy array
        embedding_np = np.expand_dims(embedding_np, axis=0)  # add an extra dimension for faiss

        # Create a unique ID for each speaker using the run_id and speaker label
        speaker_id = int(hashlib.sha1(f"{run_id}_{speaker}".encode()).hexdigest(), 16) % (10 ** 8)
        logging.info(f"Speaker {speaker} has ID {speaker_id} // {run_id}")
        if len(speaker_list) > 0:
            # Add the speaker to the speaker map
            speaker_map[str(speaker_id)] = speaker_list[count]

        if db.ntotal > 0:  # if the database is not empty
            # compute the cosine distance between the new embedding and all existing embeddings in the database
            distances, _ = db.search(embedding_np, db.ntotal) # TODO: Do we need all of these?
            
            logging.info(f"{speaker} min distance: {np.min(distances)}")
            #########
            # Search for similar embeddings
            THRESHOLD = 50_000
            r = db.range_search(embedding_np, THRESHOLD)
            num_results = r[0][1]
            id_list = list(r[2])
            
            # Get the speaker name from the speaker map
            identified_speakers = set()
            for _id in id_list:
                identified_speakers.add(speaker_map.get(str(_id), "Unknown"))

            logging.info(f"Identified speakers: {identified_speakers} across {num_results} entries.")
            if len(identified_speakers) > 0 and list(identified_speakers)[0] == speaker_list[count]:
                logging.info("Speaker DB agrees with provided label.")
            ###########

            if np.min(distances) < 0.01:  # if the minimum distance is less than 0.01, the exact embedding exists
                logging.info(f"Speaker {speaker} already exists in the database.")
                count += 1
                continue
            else:
                logging.info(f"Speaker {speaker} embedding is new. Adding to the database.")
            
            

        db.add_with_ids(embedding_np, np.array([speaker_id]))  # add the embedding to the database with the unique speaker_id
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
        for start_time, end_time in segments:
            segment_audio = original_audio[start_time:end_time]

            # Generate transcript for segment_audio
            # Use openai model
            temp_filename = "/tmp/temp_audio.mp3"
            
            segment_audio.export(temp_filename, format="mp3")
            
            # Skip if temp_filename is more than 20MB
            # TODO: don't skip; should be able to handle this with mp3s
            if os.path.getsize(temp_filename) > 20000000:
                print(f"Skipping segment {start_time}-{end_time} for speaker {speaker_label} because it is too large.")
                continue

            with open(temp_filename, 'rb') as audio_file:
                output = openai.Audio.transcribe("whisper-1", audio_file)
            
            # Add to speaker's transcript
            if speaker_label not in speakers_text:
                speakers_text[speaker_label] = output['text']
            else:
                speakers_text[speaker_label] += output['text'] + " "

            # Add audio segments
            if speaker_label not in speakers_audio:
                speakers_audio[speaker_label] = segment_audio
            else:
                speakers_audio[speaker_label] += segment_audio

    for speaker_label, speaker_audio in speakers_audio.items():
        # Output transcript
        transcript_path = f'{run_id}/{speaker_label}-TRANSCRIPT.txt'
        logging.info(f"Exporting transcript for speaker {speaker_label} to {transcript_path}")
        with open(transcript_path, "w") as f:
            f.write(speakers_text[speaker_label])

        # Output audio
        output_path = f'{run_id}/{speaker_label}.mp3'
        logging.info(f"Exporting segment for speaker {speaker_label} to {output_path}")
        # speaker_audio.set_channels(1)
        # speaker_audio.set_frame_rate(16000)
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
        speaker_list = args.speakers.split(',')
    except:
        speaker_list = []
        
    
    # Create a unique identifier for this run
    
    run_id = str(uuid.uuid4()) 

    # Create a directory for this run
    os.mkdir(run_id)
    
    logging.info(f"Using Run ID: {run_id}")

    # Use pyannote to extract one sample per speaker
    logging.info("Starting audio processing...")
    
    logging.info("Downloading audio file from URL...")
    # URL is arbitrary limitation due to APIs inability to process >20MB files
    r = requests.get(audio_path)
    with open(f"{run_id}/input.mp3", 'wb') as f:
        f.write(r.content)
    
    # TODO: mp3 vs wav
    
    diarize_audio(run_id, audio_path)

    extract_all_samples_per_speaker(run_id)

    process_segments_json(run_id, speaker_list)
    
    
    # TODO: Youtube download / extract / convert to wav
    # TODO: Speaker detection against existing database using embeddings from segment.json
