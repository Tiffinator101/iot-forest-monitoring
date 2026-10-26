#define HELTEC_WIRELESS_STICK
#include <heltec_unofficial.h>
#include <WiFi.h>
#include "secrets.h"

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

void setup() {
  heltec_setup();
  // Display doesn't render correctly on this board (known hardware issue), turn off Vext to save power since it's unused
  heltec_display_power(false);
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
  
  // Setup wifi
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

void loop() {
  heltec_loop();

  // If a packet was received, display it and the RSSI and SNR
  if (rxFlag) {
    rxFlag = false;
    radio.readData(rxdata);
    if (_radiolib_status == RADIOLIB_ERR_NONE) {
      Serial.printf("RX [%s]\n", rxdata.c_str());
      Serial.printf("RSSI: %.2f dBm\n", radio.getRSSI());
      Serial.printf("SNR: %.2f dB\n", radio.getSNR());
    }
    RADIOLIB_OR_HALT(radio.startReceive(RADIOLIB_SX126X_RX_TIMEOUT_INF));
  }
}

// Can't do Serial or display things here, takes too much time for the interrupt
void rx() {
  rxFlag = true;
}
