#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
from pydub import AudioSegment
from queue import PriorityQueue
import numpy as np
import replicate
import argparse
import requests
import chromadb
import logging
import whisper
import sqlite3
import hashlib
import random
import faiss
import tqdm
import json
import uuid
import os

class DefaultList(list):
    def __init__(self, default_value, *args):
        self.default_value = default_value
        super().__init__(*args)

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except IndexError:
            return self.default_value

def getSpeakerId(run_id, speaker):
    return int(hashlib.sha1(f"{run_id}_{speaker}".encode()).hexdigest(), 16) % (10 ** 8)

def insert_speaker_embeddings_to_faiss_db(embeddings, run_id, speaker_list):
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
    # TODO: Replace with db calls
    try:
        with open('speaker_map.json', 'r') as f: # TODO: Make configurable
            speaker_map = json.load(f)
    except:
        logging.warning("First run? Speaker map not found. Creating new speaker map.")
        speaker_map = {}

    count = 0
    for speaker, embedding in embeddings.items():
        # Create a unique ID for each speaker using the run_id and speaker label
        speaker_id = getSpeakerId(run_id, speaker)
        logging.info(f"Speaker {speaker} has ID {speaker_id} // {run_id}")

        speaker_label = f"Unknown-{speaker_id}"
        if len(speaker_list) > 0:
            speaker_label = speaker_list[count] # User has provided them in order
            logging.info(f"Processing labeled speaker {speaker_label}")
            

        embedding_np = np.asarray(embedding).astype('float32')  # convert the embedding to numpy array
        embedding_np = np.expand_dims(embedding_np, axis=0)  # add an extra dimension for faiss

        id_list = []
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
        
        # Add speaker to the database
        cursor.execute("INSERT INTO speakers (id, name, run_id) VALUES (?, ?, ?)", (speaker_id, speaker_label, run_id))
        conn.commit()

        # Update the speaker map in the database
        speaker_names = set([speaker_map[str(i)] for i in id_list])
        # Filter out anything starting with "Unknown"
        speaker_names = [name for name in speaker_names if not name.startswith("Unknown")]
        # If we have one result, that's the name to replace all other unknowns with
        if len(speaker_names) == 1:
            # Update instances of the speaker map from id_list, using the name
            for i in id_list:
                cursor.execute("UPDATE speakers SET name = ? WHERE id = ?", (speaker_names[0], i))
            conn.commit()
            logging.info(f"Updated {len(id_list)} entries with speaker name {speaker_names[0]}")
        elif len(speaker_names) > 1:
            logging.warning(f"Found multiple speakers: {speaker_names}. This requires a deconflict.")
            
        count += 1

    faiss.write_index(db, db_file_path)  # save the database to a file
    
    # Write the speaker_map back to file
    with open('speaker_map.json', 'w') as f:
        json.dump(speaker_map, f)

    return db
    
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


    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Define a lock object to synchronize threads
    lock = threading.Lock()
    
    for speaker_label, segments in tqdm.tqdm(speakers_segments.items()):
        # Create thread-safe queues for the results
        speakers_text_queue = PriorityQueue()
        speakers_audio_queue = PriorityQueue()
        # Define a function to process each segment
        def process_segment(segment):
            if os.getenv('USE_LOCAL_WHISPER') == "1":
                logging.info("Using local whisper model.")
                model = whisper.load_model("base")
            start_time, end_time = segment
            segment_audio = original_audio[start_time:end_time]

            # Generate transcript for segment_audio
            # Generate a random number or other identifier to allow for running in parallel
           
            temp_filename = f"/tmp/temp_audio_{random.randint(1, 100000)}.mp3"
            
            segment_audio.export(temp_filename, format="mp3")
            duration_seconds = segment_audio.duration_seconds
            # Skip if temp_filename is more than 20MB
            # TODO: don't skip; should be able to handle this with mp3s
            #logging.info(f"Processing segment of {duration_seconds} seconds.")
            if os.path.getsize(temp_filename) > 20000000:
                logging.warn(f"Skipping segment {start_time}-{end_time} for speaker {speaker_label} because it is too large.")
                return

            try:
                if os.getenv('USE_LOCAL_WHISPER') == "1":
                    output = model.transcribe(temp_filename, fp16=False, language='English')
                else:
                    with open(temp_filename, 'rb') as audio_file:
                        output = replicate.run("daanelson/whisperx:9aa6ecadd30610b81119fc1b6807302fd18ca6cbb39b3216f430dcf23618cedd",
                        input={"audio": audio_file})
                        output = output[0]

            except Exception as e:
                logging.error(f"Error transcribing segment {start_time}-{end_time} for speaker {speaker_label}: {e}")
            
            conn_local = sqlite3.connect(DB_PATH)
            cursor_local = conn_local.cursor()
   
            # Add to segments table
            cursor_local.execute("""
                INSERT INTO segments (run_id, speaker_id, start_time, end_time, transcript)
                VALUES (?, ?, ?, ?, ?)
            """, (run_id, getSpeakerId(run_id, speaker_label), start_time, end_time, output['text']))
            conn_local.commit()
            # Perform database operations
            cursor_local.close()
            conn_local.close()

             # Add to speaker's transcript
            
            # Using start_time will automatically sort in order
            # These are thread safe
            speakers_text_queue.put((start_time, (speaker_label, start_time, output['text'] + " ")))
            speakers_audio_queue.put((start_time, (speaker_label, start_time, segment_audio)))

        # Use a ThreadPoolExecutor to run the process_segment function in parallel for each segment
        
        MAX_WORKERS = 16 if os.getenv('USE_LOCAL_WHISPER') == "1" else 8
        logging.info(f"Using {MAX_WORKERS} workers to transcribe.")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_segment, segments))
            executor.shutdown(wait=True)
        
        # This works; above there's an issue with stitching things back together?
        # for segment in segments:
        #     process_segment(segment)

    speakers_audio = {}
    speakers_text = {}
    # Rebuild the audio and text for each speaker
    # These should be in order, so we just distrbute to the right speaker
    while not speakers_audio_queue.empty():
        _, (speaker_label, start_time, audio) = speakers_audio_queue.get()
        if speaker_label not in speakers_audio:
            speakers_audio[speaker_label] = audio
        else:
            speakers_audio[speaker_label] += audio
    
    while not speakers_text_queue.empty():
        _, (speaker_label, start_time, text) = speakers_text_queue.get()
        if speaker_label not in speakers_text:
            speakers_text[speaker_label] = [(start_time, text)]
        else:
            speakers_text[speaker_label].append((start_time, text))
    
    for speaker_label, speaker_audio in speakers_audio.items():
        # Sort the transcripts by start_time to ensure they are in order
        speakers_text[speaker_label+"_fulltext"] = " ".join([text for _, text in speakers_text[speaker_label]])
        
        # Output transcript
        transcript_path = f'{run_id}/{speaker_label}-TRANSCRIPT.txt'
        logging.info(f"Exporting transcript for speaker {speaker_label} to {transcript_path}")

        with open(transcript_path, "w") as f:
            f.write(speakers_text[speaker_label+"_fulltext"])

        # Output audio
        output_path = f'{run_id}/{speaker_label}.mp3'
        logging.info(f"Exporting segment for speaker {speaker_label} to {output_path}")
        speaker_audio.export(output_path, format="mp3")

        speaker_id = getSpeakerId(run_id, speaker_label)

        logging.info("Generating embeddings...")
        
        chroma_collection = chroma.get_or_create_collection("transcript_embeddings")
        # Embed and store
        chroma_collection.add(
              ids = [f"{run_id}_{speaker_id}_{start}" for start, _ in speakers_text[speaker_label]]
            , documents = [text for _, text in speakers_text[speaker_label]]
            , metadatas = [{'run_id': run_id, 'speaker_id': speaker_id, 'speaker_label': speaker_label, 'start': start} for start, _ in speakers_text[speaker_label]]
            #, embeddings = [list(x) for x in embeddings], # chroma embeds these automatically
        )  
        
    logging.info("All audio segments processed.")
    
def diarize_audio(run_id, audio_path):
 
    logging.info("Starting diarization...")

    # Use replicate as the hosted diarization solution

    output_location = replicate.run(
        "lucataco/speaker-diarization:718182bfdc7c91943c69ed0ac18ebe99a76fdde67ccd01fced347d8c3b8c15a6",
        input={"audio": audio_path}
        #input={"audio": open(audio_path, "rb")} # This errors out with larger files (typically > 20MB)
    )
    
    logging.info("Diarization complete.")

    # Replicate saves the results elsewhere
    r = requests.get(output_location)
    # Save output to run_id folder
    diarize_output = r.json()
    with open(f'{run_id}/segments.json', "w") as f:
        json.dump(diarize_output, f)
    
    for elt in diarize_output['segments']:
        speaker = elt['speaker']
        start = elt['start']
        stop = elt['stop']
        speaker_id = getSpeakerId(run_id, speaker)
        cursor.execute("INSERT INTO diarize_segments (run_id, speaker, speaker_id, start, stop) VALUES  (?,?,?,?,?)", (run_id, speaker, speaker_id, start, stop))

    for speaker in diarize_output['speakers']['embeddings'].keys():
        speaker_id = getSpeakerId(run_id, speaker)
        embeddings = diarize_output['speakers']['embeddings'][speaker]
        cursor.execute("INSERT INTO diarize_embeddings (run_id, speaker, speaker_id, embeddings) VALUES  (?,?,?,?)", (run_id, speaker, speaker_id, str(embeddings)))
        # TODO: proper embeddings storage

    conn.commit()

    return output_location

def create_schema(DB_PATH):
    # Open a new sqlite3 connection
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            input_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Create the schema
    cursor.execute("""
        CREATE TABLE speakers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            run_id TEXT NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE segments ( 
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker_id INTEGER NOT NULL,
            start_time INTEGER NOT NULL,
            end_time INTEGER NOT NULL,
            transcript TEXT NOT NULL,
            run_id TEXT NOT NULL
            --FOREIGN KEY (speaker_id) REFERENCES speakers (id)
            --FOREIGN KEY (run_id) REFERENCES runs (run_id)
        ); 
    """)

    cursor.execute("""
        CREATE TABLE diarize_segments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker TEXT NOT NULL,
            speaker_id INTEGER NOT NULL,
            run_id TEXT NOT NULL,
            start TEXT NOT NULL,
            stop TEXT NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE diarize_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker INTEGER NOT NULL,
            speaker_id INTEGER NOT NULL,
            run_id TEXT NOT NULL,
            embeddings TEXT NOT NULL
        );
        """)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    load_dotenv()
    # TODO: Use https://github.com/asg017/sqlite-vss ?
    
    # Check if the database file exists
    
    DB_PATH = os.getenv('DB_PATH')
    if not DB_PATH:
        logging.error("DB_PATH not set. Using default speakers.db")
        DB_PATH = "speakers.db"

    if not os.path.isfile(DB_PATH):
        # If the database file doesn't exist, create it and initialize the schema
        create_schema(DB_PATH)
    
    # Chromadb
    CHROMA_PATH = os.getenv('CHROMA_PATH')
    if not CHROMA_PATH:
        logging.error("CHROMA_PATH not set. Using default chroma")
        CHROMA_PATH = "chroma"
   
    global chroma
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    global cursor, conn
    # Open a new sqlite3 connection
    conn = sqlite3.connect(DB_PATH)
    # Create a global cursor
    cursor = conn.cursor()

    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the necessary command-line arguments
    parser.add_argument('-i', '--input', help='Path to the audio file')
    parser.add_argument('-v', '--verbose', help='Turn on verbose output of ffmpeg and whisper', action='store_true')
    parser.add_argument('-s', '--speakers', help='Comma separated list of speakers in order of appearance', type=str)
    parser.add_argument('-ll', '--log', help='Set the log level to INFO or DEBUG; is off by default', type=str)

    global args

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.log:
        log_level = logging.DEBUG if args.log == "DEBUG" else logging.INFO
        logging.basicConfig(level=log_level, format='%(asctime)s | %(levelname)s | %(message)s')
    else:
        logging.basicConfig(level=logging.NOTSET, format='%(asctime)s | %(levelname)s | %(message)s')

    # Set the input parameters
    audio_path = args.input
    
    try:
        speaker_list = DefaultList('Unknown', [x.strip() for x in args.speakers.split(',')])
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
    original_audio_path = audio_path
    if "youtube.com" in audio_path:
        
        original_audio_path = audio_path
        logging.info("Downloading from YouTube...")
        ENDPOINT = os.getenv("YT_ENDPOINT")
        full_url = ENDPOINT + audio_path
        audio_path = requests.get(full_url).text
        logging.info(f"Using generated audio URL {audio_path}")

        # Check if the youtube link is already in the runs database
        cursor.execute("SELECT * FROM runs WHERE input_path=?", (original_audio_path,))
        if cursor.fetchone() is not None:
            logging.warning("The youtube link is already in the database.")
            #exit(1)

    r = requests.get(audio_path)
    with open(f"{run_id}/input.mp3", 'wb') as f:
        f.write(r.content)
    
    
    # Insert a row of data
    cursor.execute("INSERT INTO runs (run_id, input_path, output_path) VALUES (?,?,?)", (run_id, original_audio_path, audio_path))
    
    # Save (commit) the changes
    conn.commit()
    
    diarize_audio(run_id, audio_path)

    extract_all_samples_per_speaker(run_id)

    with open(f'{run_id}/segments.json', 'r') as f:
        data = json.load(f)

    embeddings = data['speakers']['embeddings']
    insert_speaker_embeddings_to_faiss_db(embeddings, run_id, speaker_list)
    
    conn.close()
