// Turns the 'PRG' button into the power button, long press is off
#define HELTEC_POWER_BUTTON   // must be before "#include <heltec_unofficial.h>"

#define HELTEC_WIRELESS_STICK

// Tells the library not to create its own 'display' object, we're doing that ourselves
#define HELTEC_NO_DISPLAY_INSTANCE

// still creates 'radio' and 'button' instances, just not 'display'
#include <heltec_unofficial.h>

SSD1306Wire display(0x3c, SDA_OLED, SCL_OLED, GEOMETRY_64_32);

void setup() {
  heltec_setup();
  Serial.println("Serial works");

  // Manually power and reset the display, same sequence the library
  // would have run, but with a longer settle delay
  heltec_ve(true);
  delay(50);
  pinMode(RST_OLED, OUTPUT);
  digitalWrite(RST_OLED, HIGH);
  delay(1);
  digitalWrite(RST_OLED, LOW);
  delay(20);
  digitalWrite(RST_OLED, HIGH);

  display.init();
  display.setContrast(255);

  // Raw draw only, no print()/println() involved
  display.clear();
  display.setColor(WHITE);
  display.fillRect(0, 0, 64, 32);
  display.display();

  Serial.println("Display init attempted, check the screen now");

  int state = radio.begin();
  if (state == RADIOLIB_ERR_NONE) {
    Serial.println("Radio works");
  } else {
    Serial.printf("Radio fail, code: %i\n", state);
  }
}

void loop() {
  heltec_loop();
  if (button.isSingleClick()) {
    Serial.println("Button works");
    for (int n = 0; n <= 100; n++) { heltec_led(n); delay(5); }
    for (int n = 100; n >= 0; n--) { heltec_led(n); delay(5); }
    Serial.println("LED works");
  }
}
