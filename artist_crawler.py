import requests
import logging
import random
import time
from urllib.parse import quote_plus, urlparse
import sys
import concurrent.futures
import threading
import json
import os
from multiprocessing import Value

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Threading lock for writing to the file and progress counter
file_lock = threading.Lock()
progress_lock = threading.Lock()


# List of User-Agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.0; rv:86.0) Gecko/20100101 Firefox/86.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
]

def rapidapi_search(query, num_results=5):
    querystring = {"q": query, "limit": str(num_results)}
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS)
    }
    
    try:
        response = requests.get(f"https://www.google.com/search?q={querystring}")
        if response.status_code != 200:
            logging.error(f"Failed to retrieve search results for {query} with status code {response.status_code}")
            return []

        # will need to modify this part for google search
        return response.json().get('data', [])

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return []

def is_valid_domain(artist_name, url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    artist_name_parts = artist_name.lower().split()

    for part in artist_name_parts:
        if len(part) >= 4 and part in domain:
            return True
    return False

def select_best_url(artist_name, search_results):
    for result in search_results:
        url = result.get('url')
        if is_valid_domain(artist_name, url):
            return url
    return None


def load_artist_names(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file]
    
def load_progress(thread_id):
    progress_file = f"progress_thread_{thread_id}.txt"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return file.read().strip()  # Return the last processed artist
    return None

def save_progress(thread_id, artist_name):
    progress_file = f"progress_thread_{thread_id}.txt"
    with open(progress_file, 'w') as file:
        file.write(artist_name)  # Save the current artist name as the last processed

def calculate_initial_progress(threads, artist_chunks):
    total_processed = 0
    for thread_id in range(threads):
        last_processed_artist = load_progress(thread_id)
        if last_processed_artist:
            try:
                chunk = artist_chunks[thread_id]
                processed_count = chunk.index(last_processed_artist) + 1
                total_processed += processed_count
            except ValueError:
                continue  # If the last processed artist is not in the chunk, skip
    return total_processed

def process_artist_chunk(artist_chunk, output_file, artist_names_file, thread_id, progress_counter, total_artists):
    last_processed_artist = load_progress(thread_id)
    
    # Skip artists until we reach the last processed one
    if last_processed_artist:
        try:
            start_index = artist_chunk.index(last_processed_artist) + 1
            artist_chunk = artist_chunk[start_index:]
        except ValueError:
            pass  # Start from the beginning if the artist is not in the chunk

    for artist_name in artist_chunk:
        search_results = rapidapi_search(artist_name)
        best_url = select_best_url(artist_name, search_results)

        if best_url:
            logging.info(f"Valid URL for {artist_name}: {best_url}")
            with file_lock:
                with open(output_file, 'a') as file:
                    file.write(f"{best_url}\n")

        # Save progress after each artist is processed
        save_progress(thread_id, artist_name)

        # Update and display progress
        with progress_lock:
            progress_counter.value += 1
            logging.info(f"Progress: {progress_counter.value}/{total_artists} artists processed")

    # Update the main artist file after processing the chunk
    with file_lock:
        with open(artist_names_file, 'r') as file:
            all_artists = [line.strip() for line in file]
        
        remaining_artists = [artist for artist in all_artists if artist not in artist_chunk]

        with open(artist_names_file, 'w') as file:
            for artist in remaining_artists:
                file.write(f"{artist}\n")

def main(artist_names_file, output_file):
    threads = 20  # Adjust the number of threads as needed
    artist_names = load_artist_names(artist_names_file)
    chunk_size = len(artist_names) // threads or 1

    # Divide artist names into chunks for each thread
    artist_chunks = [artist_names[i:i + chunk_size] for i in range(0, len(artist_names), chunk_size)]

    # Initialize progress counter with the initial progress
    initial_progress = calculate_initial_progress(threads, artist_chunks)
    progress_counter = Value('i', initial_progress)
    total_artists = len(artist_names)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_artist_chunk, chunk, output_file, artist_names_file, i, progress_counter, total_artists) for i, chunk in enumerate(artist_chunks)]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python artist_crawler.py <artist_names_file> <output_file>")
        sys.exit(1)
    
    artist_names_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(artist_names_file, output_file)
