#include "esp_camera.h"
#include "board_config.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Preferences.h>
#include "secrets.h"

// SD Module SPI Wiring
#define SD_SCK  17
#define SD_MOSI 15
#define SD_MISO 16
#define SD_CS   9

// WiFi connection timeout
#define WIFI_TIMEOUT_MS 10000

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Starting camera...");

  camera_config_t config = {};   // = {} zeroes every field, so any field not set below (e.g. jpeg_buffer_size) defaults to 0 instead of leftover memory
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }
  Serial.println("Camera ready.");

  // Wi-Fi section
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

  // MicroSD Card Section
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);

  if (!SD.begin(SD_CS)) {
    Serial.println("Card Mount Failed");
    return;
  }

  if (SD.cardType() == CARD_NONE) {
    Serial.println("No SD card attached");
    return;
  }

  Serial.println("Card ready.");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("Capture failed");
  }
  else {
    Serial.printf("Captured successfully!");

    // Non-volatile counter for filename for micro SD card
    Preferences preferences;
    preferences.begin("my-firebeetle", false);
    unsigned int counter = preferences.getUInt("counter", 1);
    char filename[20];
    sprintf(filename, "/Capture%04u.jpg", counter);
    counter++;
    preferences.putUInt("counter", counter);
    preferences.end();

    // Write the image to the micro SD card instead
    File file = SD.open(filename, FILE_WRITE);
    if (!file) {
      Serial.println("Failed to open file for writing");
    }
    else {
      file.write(fb->buf, fb->len);
      file.close();
    }

    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(SERVER_URL);
      http.addHeader("Content-Type", "application/octet-stream");

      int httpResponseCode = http.POST(fb->buf, fb->len);

      if (httpResponseCode == 200) {
        Serial.println(httpResponseCode);
      }
      else{
        Serial.println("Error sending image");
      }

      http.end();  //Free resources
    }
    else{
      Serial.println("Error in WiFi connection");
    }

    esp_camera_fb_return(fb);
  }

  delay(5000);  // Repeat every 5 seconds
}
