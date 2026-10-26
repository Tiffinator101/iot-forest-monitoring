import database
from datetime import datetime
from paho.mqtt import client as mqtt_client
from paho.mqtt.enums import CallbackAPIVersion

# MQTT configuration
MQTT_SERVER = "127.0.0.1"   # localhost is used here because the MQTT broker is running on the same machine as the subscriber
MQTT_PORT = 1883
MQTT_TOPIC = "treemonitor/sensor_readings"
CLIENT_ID = "BaseStationSubscriber"

# Define a dictionary to map the sensor reading keys to their corresponding field names and types
SENSOR_READINGS_FIELDS = {
    "INDEX": ("packet_index", int),
    "H": ("air_humidity", float),
    "T": ("air_temperature", float),
    "ADC": ("soil_moisture_adc", int),
    "BAT": ("battery_adc", int),
    "RSSI": ("rssi", float),
    "SNR": ("snr", float)
}

def connect_mqtt():
    # Define the callback functions for MQTT events
    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(MQTT_TOPIC)
        else:
            print(f"Failed to connect, return code {reason_code}")

    def on_message(client, userdata, msg):
        decoded_payload = msg.payload.decode(errors="replace")
        print(f"Received `{decoded_payload}` from `{msg.topic}` topic")
        process_sensor_reading(decoded_payload)

    # Create an MQTT client instance
    client = mqtt_client.Client(
        client_id=CLIENT_ID,
        callback_api_version=CallbackAPIVersion.VERSION2,
    )

    # Set the callback functions for connection and message events and connect to the MQTT broker
    client.on_message = on_message
    client.on_connect = on_connect
    client.connect(MQTT_SERVER, MQTT_PORT)
    return client

def run():
    database.create_tables()
    client = connect_mqtt()
    client.loop_forever()

def parse_sensor_reading_message(message):
    # Note: Below shows the expected format of the sensor reading message that will be received from the MQTT broker.
    # Payload format: ID<node> INDEX:<n> H:<humidity> T:<temp> ADC:<adc> BAT:<battery> RSSI:<rssi> SNR:<snr>
    # Example:        ID012267 INDEX:0 H:64.34 T:24.69 ADC:874 BAT:929 RSSI:-23.00 SNR:11.25
    tokens = message.split()
    node_id = tokens[0]
    key_value_pairs = tokens[1:] # The first token is "ID<node>", so we skip it and only keep the key-value pairs starting from the second token
    sensor_data = {}
    for pair in key_value_pairs:
        key, value = pair.split(":", 1)
        sensor_data[key] = value
    return node_id, sensor_data

def format_sensor_readings(sensor_data):
    formatted_readings = {}
    for key, (field_name, field_type) in SENSOR_READINGS_FIELDS.items():
        formatted_readings[field_name] = field_type(sensor_data[key])
    return formatted_readings

def process_sensor_reading(payload):
    try:
        # Parse the sensor reading message to extract the node ID and sensor data
        node_id, sensor_data = parse_sensor_reading_message(payload)
        # Format the sensor readings to match the database schema
        formatted_readings = format_sensor_readings(sensor_data)
    except (ValueError, IndexError, KeyError) as e:
        print(f"Error parsing and formatting the sensor reading. Payload: {payload}, Error: {e}")
        return
    # Get the current timestamp in the format "YYYY-MM-DD HH:MM:SS"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Get the tree ID based on the node ID (the node-to-tree dictionary is in database.py)
    tree_id = database.SENSOR_TREE_IDS.get(node_id)
    if tree_id is None:
        print(f"Unknown node ID: {node_id}. Storing reading without a tree ID.")
    # Extract the individual sensor readings from the formatted readings dictionary
    packet_index = formatted_readings["packet_index"]
    air_humidity = formatted_readings["air_humidity"]
    air_temperature = formatted_readings["air_temperature"]
    soil_moisture_adc = formatted_readings["soil_moisture_adc"]
    battery_adc = formatted_readings["battery_adc"]
    rssi = formatted_readings["rssi"]
    snr = formatted_readings["snr"]
    # Insert the sensor reading record into the database
    database.insert_sensor_reading_record(timestamp, node_id, tree_id, packet_index, air_humidity, air_temperature, soil_moisture_adc, battery_adc, rssi, snr)

if __name__ == "__main__":
    run()
