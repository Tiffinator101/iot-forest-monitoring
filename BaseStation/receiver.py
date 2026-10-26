from flask import Flask, request
from datetime import datetime
from pathlib import Path
import database

app = Flask(__name__)
MAX_IMAGE_SIZE_BYTES = 1024 * 1024     # 1 MB in bytes
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE_BYTES

@app.route("/upload", methods=["POST"])
def upload_image():
    # Get the raw bytes of the uploaded image
    image_raw_bytes = request.get_data()
    # Defining the filename and path to save the image
    now = datetime.now()   # Take the time once, so the filename and the database row always match
    file_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")   # format = year-month-day_hour-minute-second, no colons since Windows does not allow them in filenames
    db_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")     # format = year-month-day hour:minute:second, the format SQLite's date functions accept and the same as sensor_readings
    camera_id = request.headers.get("Camera-ID")         # Get the camera ID from the request headers
    if not camera_id:
        camera_id = "TreeCamUnknown"   # If the camera ID is not provided in the request headers, set it to "TreeCamUnknown"
        print("Camera ID is missing in the request headers.")
    filename = f"{camera_id}_{file_timestamp}.jpg"
    file_path = Path(__file__).parent/"images"/filename   # Path(__file__).parent gives the directory of receiver.py
    failed_file_path = Path(__file__).parent/"failed"/filename   # Defines a path for the images that failed to be inserted into the database
    try:
        # Create the images directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)       # Here file_path.parent is the image's directory path
        # Create the failed images directory if it doesn't exist
        failed_file_path.parent.mkdir(parents=True, exist_ok=True)  # Here failed_file_path.parent is the failed image's directory path
        # Save the image to the specified path
        file_path.write_bytes(image_raw_bytes)
        # Insert image metadata into the database
        image_id = database.insert_image_record(db_timestamp, camera_id, file_path)
        if image_id is None:
            file_path.replace(failed_file_path)  # Move the image to the failed directory if the database insertion fails
            return "Failed to insert image record into the database.", 500   # 500 is the HTTP response status codes for server error responses
        # image_id is not None, meaning the image record was successfully inserted into the database
        return f"Image uploaded successfully with ID: {image_id}", 200   # 200 is the HTTP response status codes for successful responses
    except OSError as e:
        return f"A failure occurred. Error: {str(e)}", 500   # 500 is the HTTP response status codes for server error responses

if __name__ == "__main__":
    # Create the database tables if they don't exist and see if it returned True (meaning the tables were created successfully)
    if database.create_tables():
        # run app in debug mode on port 5000
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("Failed to create database tables.")
