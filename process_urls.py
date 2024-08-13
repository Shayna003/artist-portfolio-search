import threading
import queue
import signal
import sys
import os
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from pymongo import MongoClient
from multiprocessing import Value
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory for screenshots
screenshot_dir = "screenshots"
#os.makedirs(screenshot_dir, exist_ok=True)

# Lock for file writing to prevent race conditions
file_lock = threading.Lock()

# Queues for different stages
screenshot_queue = queue.Queue()

# Global flag to indicate termination request
terminate_flag = threading.Event()

# Condition variable for waiting and notifying
screenshot_condition = threading.Condition()

total_urls = 0
screenshots_captured = Value('i', 0)
screenshots_analyzed = Value('i', 0)

# Function to handle program termination and save progress
def signal_handler(sig, frame):
    print('Termination signal received. Saving progress...')
    terminate_flag.set()  # Signal threads to stop
    with screenshot_condition:
        screenshot_condition.notify_all()  # Wake up all threads
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Function to capture screenshot of a given URL
def capture_screenshot(driver, url, save_path):
    try:
        print(f"Fetching URL: {url}")
        driver.get(url)

        # Wait for the body tag to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        print(f"Body tag present for URL: {url}")

        # Wait until the page is completely loaded
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )
        print(f"Page fully loaded for URL: {url}")

        screenshot_path = os.path.join(save_path, f"{urlparse(url).netloc}.png")
        time.sleep(10)  # Consider reducing this sleep time if unnecessary
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot captured for URL: {url}")
        return screenshot_path

    except Exception as e:
        print(f"Error capturing screenshot for {url}: {e}")
        return None

# Function for screenshot capture worker
def screenshot_capture(thread_id, urls):
    # Setup WebDriver (Firefox in this case)
    options = Options()
    options.headless = True
    options.add_argument("--headless")
    options.add_argument("--window-size=1024x768")
    options.page_load_strategy = 'eager'  # Eager means the driver will wait for the DOMContentLoaded event
    service = Service(executable_path="../geckodriver")  # Replace with actual path to geckodriver
    driver = webdriver.Firefox(service=service, options=options)

    # Load progress for this thread
    last_processed_index = load_progress(thread_id)

    for index, url in enumerate(urls):
        if terminate_flag.is_set():
            break

        if index < last_processed_index:
            continue  # Skip already processed URLs

        screenshot_path = capture_screenshot(driver, url, screenshot_dir)
        if screenshot_path:
            with screenshot_condition:
                screenshot_queue.put((url, screenshot_path))
                screenshot_condition.notify()  # Notify an analyzer thread
            screenshots_captured.value += 1  # Increment counter

            # Save the captured screenshot to a file and add it to queue
            with file_lock:
                with open("screenshots_gathered.txt", 'a') as f:
                    f.write(f"{url}\n")

        # Save progress after each URL is processed
        save_progress(thread_id, index + last_processed_index)

    driver.quit()

# Function to extract color profile from an image
def extract_color_profile(image_path, num_colors=5):
    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        return []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image {image_path}.")
        return []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    kmeans = KMeans(n_clusters=num_colors)
    try:
        kmeans.fit(image)
    except ValueError:
        print(f"Not enough colors in image {image_path} for clustering.")
        return []
    
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    counts = np.bincount(labels)
    percentages = counts / len(labels)
    
    # Filter out colors with percentage less than 1%
    color_profile = [
        {"color": [int(c) for c in colors[i]], "percentage": float(percentages[i])}
        for i in range(num_colors)
        if i < len(percentages) and percentages[i] >= 0.01
    ]
    
    # Sort color profile by percentage in descending order
    color_profile.sort(key=lambda x: x["percentage"], reverse=True)
    
    return color_profile

# Function to store data into MongoDB
def store_data(url, screenshot_path, color_profile):
    client = MongoClient('localhost', 27017)
    db = client['portfolio_db']
    collection = db['portfolios']
    
    data = {
        "url": url,
        "screenshot": screenshot_path,
        "color_profile": color_profile,
        "rating": 0  # Set an initial rating value
    }
    collection.insert_one(data)

# Function for screenshot analyzer worker
def screenshot_analyzer(thread_id):
    while not terminate_flag.is_set():
        with screenshot_condition:
            while screenshot_queue.empty() and not terminate_flag.is_set():
                screenshot_condition.wait()  # Wait until there are items in the queue or termination signal

            if terminate_flag.is_set():
                break

            try:
                url, screenshot_path = screenshot_queue.get_nowait()  # Non-blocking get from the queue
            except queue.Empty:
                continue

        color_profile = extract_color_profile(screenshot_path)
        store_data(url, screenshot_path, color_profile)
        screenshots_analyzed.value += 1  # Increment counter

        # Save the analyzed screenshot to a file
        with file_lock:
            with open("screenshots_analyzed.txt", 'a') as f:
                f.write(f"{url}\n")

        screenshot_queue.task_done()

# Function to save the progress of each capture thread
def save_progress(thread_id, index):
    progress_file = f"progress/progress_thread_{thread_id}.txt"
    with open(progress_file, 'w') as file:
        file.write(str(index))  # Save the last processed index

# Function to load the progress of each capture thread
def load_progress(thread_id):
    progress_file = f"progress/progress_thread_{thread_id}.txt"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return int(file.read().strip())
    return 0  # Start from the beginning

# Function to save progress for all threads
def save_all_progress():
    for thread_id in range(1, 21):  # 20 capture threads
        progress_file = f"progress/progress_thread_{thread_id}.txt"
        with open(progress_file, 'w') as file:
            file.write("Progress saved")

# Function to load existing screenshots and add them to the queue
def load_existing_screenshots():
    captured_screenshots = set()
    analyzed_screenshots = set()

    # Load captured screenshots
    if os.path.exists('screenshots_gathered.txt'):
        with open('screenshots_gathered.txt', 'r') as file:
            captured_screenshots = set(line.strip() for line in file if line.strip())

    # Load analyzed screenshots
    if os.path.exists('screenshots_analyzed.txt'):
        with open('screenshots_analyzed.txt', 'r') as file:
            analyzed_screenshots = set(line.strip() for line in file if line.strip())

    # Add screenshots that have been captured but not analyzed to the queue
    to_analyze = captured_screenshots - analyzed_screenshots
    for url in to_analyze:
        screenshot_path = os.path.join(screenshot_dir, f"{urlparse(url).netloc}.png")
        screenshot_queue.put((url, screenshot_path))

    # Update counters based on loaded data
    screenshots_captured.value = len(captured_screenshots)
    screenshots_analyzed.value = len(analyzed_screenshots)
    
def display_progress():
    while not terminate_flag.is_set():
        time.sleep(10)
        print(f"Screenshots captured: {screenshots_captured.value}/{total_urls}")
        print(f"Screenshots analyzed: {screenshots_analyzed.value}/{total_urls}")


if __name__ == "__main__":
    # Load existing screenshots that need to be analyzed
    load_existing_screenshots()

    # Read initial URLs from a file and divide into chunks for each thread
    with open('urls_to_process.txt', 'r') as f:
        urls = f.readlines()

    total_urls = len(urls)

    # Divide URLs into chunks for each capture thread
    chunk_size = total_urls // 20
    url_chunks = [urls[i:i + chunk_size] for i in range(0, total_urls, chunk_size)]

    # Start the screenshot capture threads
    for i in range(20):
        threading.Thread(target=screenshot_capture, args=(i + 1, url_chunks[i]), daemon=True).start()
    
    # Start the screenshot analyzer threads
    for i in range(10):
        threading.Thread(target=screenshot_analyzer, args=(i + 1,), daemon=True).start()

    # Start progress display thread
    threading.Thread(target=display_progress, daemon=True).start()

    print("Waiting until all tasks are done...")

    # Wait until all screenshots are analyzed
    while screenshots_analyzed.value < total_urls:
        time.sleep(1)  # Sleep for a short time to avoid busy-waiting

    print("All tasks completed.")