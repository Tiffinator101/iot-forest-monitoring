#include <Arduino.h>
#include "esp_camera.h"
#include "board_config.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Preferences.h>
#include "secrets.h"

// Camera node identification
#define CAMERA_ID "TreeCam1"

// SD Module SPI Wiring
#define SD_SCK  17
#define SD_MOSI 15
#define SD_MISO 16
#define SD_CS   9

// WiFi connection timeout
#define WIFI_TIMEOUT_MS 10000

// Settle Camera Exposure
#define SETTLE_FRAMES 20

// Deep sleep
#define uS_TO_S_FACTOR 1000000ULL   // Conversion factor for micro seconds to seconds
#define TIME_ASLEEP_FOR 60        // this is how long the camera module is alseep for before it has to wake up again (in seconds)

void setup() {
  Serial.begin(115200);
  delay(1000);
  // Print the camera ID to make it easier to identify
  Serial.println(CAMERA_ID);
  // Only continue if the camera is configured because without it, nothing can be captured
  if (configureCamera()) {
    configureWifi();
    configureMicroSD();
    // After configuration is complete, the settling of the exposure and capture cycle can start
    settleCameraExposure();
    runCaptureCycle();
  }
  // Once that is done, it can go into deep sleep
  deepSleep();
}

bool configureCamera() {
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
  config.frame_size = FRAMESIZE_QXGA;
  config.jpeg_quality = 10;
  config.fb_count = psramFound() ? 2 : 1;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_hmirror(s, 1);    // flip horizontally so left and right match the real scene
  s->set_wb_mode(s, 1);    // sunny white balance preset
  s->set_ae_level(s, -1);  // slightly darker exposure target to stop the highlights clipping
  s->set_lenc(s, 1);       // lens shading correction
  Serial.println("Camera ready.");
  return true;
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

void configureMicroSD() {
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

void settleCameraExposure() {
  for (int frame = 1; frame <= SETTLE_FRAMES; frame++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb != NULL) {
      esp_camera_fb_return(fb);
    }
  }
  Serial.println("Camera exposure settled.");
}

void runCaptureCycle() {
  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("Capture failed");
    return;
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

    // Always write the image to the micro SD card as a backup, before any upload is attempted
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
      // Add a header to show the camera ID
      http.addHeader("Camera-ID", CAMERA_ID);

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
}

void printWakeUpReason() {
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_TIMER) {
    Serial.println("Wakeup caused by timer");
  }
  else {
    Serial.println("Not caused by deep sleep");
  }
}

void deepSleep() {
  printWakeUpReason();
  esp_sleep_enable_timer_wakeup(TIME_ASLEEP_FOR * uS_TO_S_FACTOR);
  Serial.println("Setup ESP32 to sleep for every " + String(TIME_ASLEEP_FOR) + " Seconds");
  Serial.println("Going to sleep now");
  Serial.flush();
  esp_deep_sleep_start();
  Serial.println("This will never be printed"); 
}

void loop() {
  // Never reached because setup() ends by calling esp_deep_sleep_start(), which resets
  // the chip instead of returning. This empty function only exists because Arduino's
  // framework requires loop() to be defined to compile.
}
