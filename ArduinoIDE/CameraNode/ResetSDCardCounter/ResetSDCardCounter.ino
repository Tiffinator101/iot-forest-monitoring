#include <Preferences.h>

void setup() {
  Serial.begin(115200);
  delay(1000);

  Preferences preferences;
  preferences.begin("my-firebeetle", false);
  preferences.remove("counter");
  Serial.println("Counter reset");
  preferences.end();
}
  
void loop() {

}