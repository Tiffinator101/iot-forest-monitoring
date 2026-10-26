import sqlite3
from pathlib import Path

database_path = Path(__file__).parent/"database.db"   # Path(__file__).parent gives the directory of database.py

# Define dictionaries to map each sensor node and each camera to the tree it is at, so that a tree's sensor readings
# and its measurements share one tree ID (used by subscriber.py and the Tree Measurement app)
SENSOR_TREE_IDS = {
    "ID012267": 1,
    "ID012280": 2
}
CAMERA_TREE_IDS = {
    "TreeCam1": 1,
    "TreeCam2": 2
}

table_sql_statements = [
    # Create the images table if it doesn't exist
    """CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY, 
            timestamp TEXT, 
            camera_id TEXT, 
            file_path TEXT
        );""",

    # Create the sensor readings table if it doesn't exist
    """CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            node_id TEXT,
            tree_id INTEGER,
            packet_index INTEGER,
            air_humidity REAL,
            air_temperature REAL,
            soil_moisture_adc INTEGER,
            battery_adc INTEGER,
            rssi REAL,
            snr REAL
        );""",

    # Create the tree measurements table if it doesn't exist
    """CREATE TABLE IF NOT EXISTS tree_measurements (
            id INTEGER PRIMARY KEY, 
            image_id INTEGER UNIQUE,
            tree_id INTEGER,
            tree_height_m REAL,
            crown_width_m REAL,
            trunk_width_mm REAL,
            crown_length_m REAL,
            height_to_crown_base_m REAL,
            trunk_width_height_m REAL
        );"""
]

def create_connection(db_file):
    try:
        # Create a database connection to the SQLite database specified by db_file
        conn = sqlite3.connect(db_file)
        # WAL mode enabled
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except sqlite3.Error as e:
        print(f"An error occurred while creating connection: {e}")
        return None

def create_tables():
    # try to create a database connection
    connection = create_connection(database_path)
    if connection is None:
        print("Failed to create database connection.")
        return

    # try to create tables
    try:
        # create a cursor
        cursor = connection.cursor()
        # execute statements
        for statement in table_sql_statements:
            cursor.execute(statement)
        # commit the changes
        connection.commit()
        print("Tables created successfully.")
        return True  # Return True if tables were created successfully

    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")
        return False  # Return False if table creation failed

    finally:
        connection.close()

def insert_image_record(timestamp, camera_id, file_path):
    # try to create a database connection
    connection = create_connection(database_path)
    if connection is None:
        print("Failed to create database connection.")
        return

    # try to insert an image record
    try:
        # create a cursor
        cursor = connection.cursor()
        # execute statements
        cursor.execute("INSERT INTO images (timestamp, camera_id, file_path) VALUES (?, ?, ?)", (timestamp, camera_id, str(file_path)))
        # commit the changes
        connection.commit()
        print("Image record inserted successfully.")
        return cursor.lastrowid  # Return the ID of the inserted image record

    except sqlite3.Error as e:
        print(f"An error occurred while inserting image record: {e}")
        return None  # Return None if insertion failed

    finally:
        connection.close()

def insert_sensor_reading_record(timestamp, node_id, tree_id, packet_index, air_humidity, air_temperature, soil_moisture_adc, battery_adc, rssi, snr):
    # try to create a database connection
    connection = create_connection(database_path)
    if connection is None:
        print("Failed to create database connection.")
        return

    # try to insert a sensor reading record
    try:
        # create a cursor
        cursor = connection.cursor()
        # execute statements
        cursor.execute("INSERT INTO sensor_readings(timestamp, node_id, tree_id, packet_index, air_humidity, air_temperature, soil_moisture_adc, battery_adc, rssi, snr) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                       (timestamp, node_id, tree_id, packet_index, air_humidity, air_temperature, soil_moisture_adc, battery_adc, rssi, snr))
        # commit the changes
        connection.commit()
        print("Sensor reading record inserted successfully.")
        return cursor.lastrowid  # Return the ID of the inserted sensor reading record

    except sqlite3.Error as e:
        print(f"An error occurred while inserting sensor reading record: {e}")
        return None  # Return None if insertion failed

    finally:
        connection.close()

def image_id_lookup(file_path):
    # try to create a database connection
    connection = create_connection(database_path)
    if connection is None:
        print("Failed to create database connection.")
        return

    # try to look up the image ID
    try:
        # create a cursor
        cursor = connection.cursor()
        # execute statements
        cursor.execute("SELECT id FROM images WHERE file_path LIKE ?", ('%' + Path(file_path).name,))
        result = cursor.fetchone()
        if result is not None:
            return result[0]  # Return the image ID if found
        else:
            print("Image not found in the database.")
            return None  # Return None if image not found

    except sqlite3.Error as e:
        print(f"An error occurred while looking up image ID: {e}")
        return None  # Return None if lookup failed

    finally:
        connection.close()

def insert_tree_measurement_record(image_id, tree_id, tree_height_m, crown_width_m, trunk_width_mm, crown_length_m, height_to_crown_base_m, trunk_width_height_m):
    # try to create a database connection
    connection = create_connection(database_path)
    if connection is None:
        print("Failed to create database connection.")
        return

    # try to insert a tree measurement record
    try:
        # create a cursor
        cursor = connection.cursor()
        # execute statements
        cursor.execute("INSERT INTO tree_measurements(image_id, tree_id, tree_height_m, crown_width_m, trunk_width_mm, crown_length_m, height_to_crown_base_m, trunk_width_height_m) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (image_id, tree_id, tree_height_m, crown_width_m, trunk_width_mm, crown_length_m, height_to_crown_base_m, trunk_width_height_m))
        # commit the changes
        connection.commit()
        print("Tree measurement record inserted successfully.")
        return cursor.lastrowid  # Return the ID of the inserted tree measurement record

    except sqlite3.IntegrityError:
        print("This photo is already in the tree_measurements table.")
        return -1  # Return -1 if insertion failed due to integrity error

    except sqlite3.Error as e:
        print(f"An error occurred while inserting tree measurement record: {e}")
        return None  # Return None if insertion failed

    finally:
        connection.close()
