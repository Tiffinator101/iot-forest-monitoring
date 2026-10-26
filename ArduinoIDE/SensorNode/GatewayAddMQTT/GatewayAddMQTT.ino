#define HELTEC_WIRELESS_STICK
#include <heltec_unofficial.h>
#include <WiFi.h>
#include "secrets.h"
#include <PubSubClient.h>

#define FREQUENCY         868.0
#define BANDWIDTH         125.0
#define SPREADING_FACTOR  9
#define CODING_RATE       7
#define OUTPUT_POWER      10
#define PREAMBLE_LEN      8

String rxdata;
volatile bool rxFlag = false;

// WiFi connection timeout
#define WIFI_TIMEOUT_MS 10000

// MQTT Related
#define MQTT_TOPIC "treemonitor/sensor_readings"
WiFiClient espClient;
PubSubClient client(espClient);
long lastReconnectAttempt = 0;

void setup() {
  heltec_setup();
  // Display doesn't render correctly on this board (known hardware issue), turn off Vext to save power since it's unused
  heltec_display_power(false);
  // Radio
  configureRadio();
  // Setup wifi
  configureWifi();
  // MQTT Setup
  lastReconnectAttempt = 0;
  client.setServer(MQTT_SERVER, MQTT_PORT);
}

void loop() {
  // Only needed for the PRG power button (HELTEC_POWER_BUTTON), which this gateway doesn't use.
  // Kept because the library expects it every loop, and it costs nothing.
  heltec_loop();
  // If a packet was received, display it and the RSSI and SNR
  if (rxFlag) {
    rxHandler();
  }
  // Make sure the the client has a connection
  handleClientConnection();
}

// Can't do Serial or display things here, takes too much time for the interrupt
void rx() {
  rxFlag = true;
}

void rxHandler() {
  rxFlag = false;
  int state = radio.readData(rxdata);
  if (state == RADIOLIB_ERR_NONE) {
    String RSSI = String(radio.getRSSI(), 2);
    String SNR = String(radio.getSNR(), 2);
    rxdata = rxdata + " RSSI:" + RSSI + " SNR:" + SNR;
    Serial.printf("RX [%s]\n", rxdata.c_str());
    // MQTT section
    if (client.connected()) {
      client.publish(MQTT_TOPIC, rxdata.c_str());
    }
    else {
      Serial.println("MQTT not connected, reading not forwarded");
    }
  }
  else {
    Serial.printf("Receive error %d, packet dropped.\n", state);
  }
  RADIOLIB_OR_HALT(radio.startReceive(RADIOLIB_SX126X_RX_TIMEOUT_INF));
}

void configureRadio() {
  Serial.println("Radio init");
  RADIOLIB_OR_HALT(radio.begin());
  // Set the callback function for received packets
  radio.setDio1Action(rx);
  // Set radio parameters
  RADIOLIB_OR_HALT(radio.setFrequency(FREQUENCY));
  RADIOLIB_OR_HALT(radio.setBandwidth(BANDWIDTH));
  RADIOLIB_OR_HALT(radio.setSpreadingFactor(SPREADING_FACTOR));
  RADIOLIB_OR_HALT(radio.setCodingRate(CODING_RATE));
  RADIOLIB_OR_HALT(radio.setOutputPower(OUTPUT_POWER));
  RADIOLIB_OR_HALT(radio.setPreambleLength(PREAMBLE_LEN));
  // Start receiving
  RADIOLIB_OR_HALT(radio.startReceive(RADIOLIB_SX126X_RX_TIMEOUT_INF));
}

void configureWifi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);
  Serial.print("Connecting to WiFi");
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < WIFI_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected. IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connection timed out.");
  }
}

void handleClientConnection() {
  if (!client.connected()) {
    long now = millis();
    if (now - lastReconnectAttempt > 5000) {
      lastReconnectAttempt = now;
      // Attempt to reconnect
      if (reconnect()) {
        lastReconnectAttempt = 0;
      }
    }
  }
  else {
    // Client connected
    client.loop();
  }
}

boolean reconnect() {
  Serial.print("Attempting MQTT connection...");
  // Create a random client ID
  String clientId = "HeltecGateway-";
  clientId += String(random(0xffff), HEX);
  // Attempt to connect
  if (client.connect(clientId.c_str())) {
    Serial.println("connected");
    return true;
  }
  else {
    Serial.print("failed, rc=");
    Serial.print(client.state());
    Serial.println(" try again in 5 seconds");
    return false;
  }
}
