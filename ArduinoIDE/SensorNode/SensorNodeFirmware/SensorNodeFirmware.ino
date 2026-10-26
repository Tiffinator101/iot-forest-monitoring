// Sensor node firmware for the Makerfabs LoRa Soil Moisture Sensor V3 (ATmega328P).
// Based on Makerfabs' Moisture_LowPower_RadioLib_v2.1 example, changed to:
//   - transmit on 868 MHz with node ID 012267
//   - set the reading interval in seconds (TRANSMIT_INTERVAL_SECONDS)
//   - send the payload as ID<node> INDEX:<n> H:<h> T:<t> ADC:<adc> BAT:<bat>
// Uses a local copy of RadioLib 6.4.2 in src/RadioLib, with unused drivers excluded in
// src/RadioLib/BuildOptUser.h so the sketch fits in the ATmega328P's flash.
// Board settings: Arduino Pro or Pro Mini, Processor ATmega328P (3.3V, 8MHz).

#include <SPI.h>
#include <Wire.h>
#include "src/RadioLib/RadioLib.h"
#include <avr/wdt.h>
#include <avr/sleep.h>
#include "I2C_AHT10.h"

// Node ID sent at the start of every payload, must match the TREE_IDS entry in subscriber.py
String node_id = String("ID") + "012267";
// String node_id = String("ID") + "012280";

// Time between readings in seconds.
// The interval is approximate, since the watchdog oscillator drifts and each wake up adds a
// little awake time, and it is rounded down to a multiple of WATCHDOG_PERIOD_SECONDS.
#define TRANSMIT_INTERVAL_SECONDS 60

// Length of one watchdog sleep, must match the 8s set by WDP3 | WDP0 in watchdog_init()
#define WATCHDOG_PERIOD_SECONDS 8

// Number of watchdog wake ups before a reading is sent. Wake ups needed = interval / 8,
// then - 1 because loop() checks count > SLEEP_CYCLE, which is true on wake up SLEEP_CYCLE + 1.
#define SLEEP_CYCLE ((TRANSMIT_INTERVAL_SECONDS / WATCHDOG_PERIOD_SECONDS) - 1)

//Lora set, must match the gateway's receiver settings or the gateway will not hear this node
#define FREQUENCY 868.0
#define BANDWIDTH 125.0
#define SPREADING_FACTOR 9
#define CODING_RATE 7
#define OUTPUT_POWER 10
#define PREAMBLE_LEN 8
#define GAIN 0

//328p
#define DIO0 2
#define DIO1 6

#define LORA_RST 4
#define LORA_CS 10

#define SPI_MOSI 11
#define SPI_MISO 12
#define SPI_SCK 13

//pin set
#define VOLTAGE_PIN A3
#define PWM_OUT_PIN 9
#define SENSOR_POWER_PIN 5
#define ADC_PIN A2

#define DEBUG_OUT_ENABLE 1

SX1276 radio = new Module(LORA_CS, DIO0, LORA_RST, DIO1);
AHT10 humiditySensor;

bool readSensorStatus = false;
int sensorValue = 0; // variable to store the value coming from the sensor
int batValue = 0;    // the voltage of battery
int count = 0;
int ADC_O_1;           // ADC Output First 8 bits
int ADC_O_2;           // ADC Output Next 2 bits
int16_t packetnum = 0; // packet counter, we increment per xmission
float temperature = 0.0;
float humidity = 0.0;

bool AHT_init()
{
    bool ret = false;
    Wire.begin();
    if (humiditySensor.begin() == false)
    {
#if DEBUG_OUT_ENABLE
        Serial.println("AHT10 not detected. Please check wiring. Freezing.");
#endif
    }

    if (humiditySensor.available() == true)
    {
        temperature = humiditySensor.getTemperature();
        humidity = humiditySensor.getHumidity();
        ret = true;
    }
    if (isnan(humidity) || isnan(temperature))
    {
#if DEBUG_OUT_ENABLE
        Serial.println(F("Failed to read from AHT sensor!"));
#endif
    }
    return ret;
}
void Lora_init()
{
    int state = radio.begin(FREQUENCY, BANDWIDTH, SPREADING_FACTOR, CODING_RATE, RADIOLIB_SX127X_SYNC_WORD, OUTPUT_POWER, PREAMBLE_LEN, GAIN);
    if (state == RADIOLIB_ERR_NONE)
    {
#if DEBUG_OUT_ENABLE
        Serial.println(F("success!"));
#endif
    }
    else
    {
#if DEBUG_OUT_ENABLE
        Serial.print(F("failed, code "));
        Serial.println(state);
#endif
        // while (true)
        //     ;
    }
}
void setup()
{
#if DEBUG_OUT_ENABLE
    Serial.begin(115200);
    Serial.println("Soil start.");
#endif
    delay(100);

    // set up Timer 1
    pinMode(PWM_OUT_PIN, OUTPUT);

    TCCR1A = bit(COM1A0);            // toggle OC1A on Compare Match
    TCCR1B = bit(WGM12) | bit(CS10); // CTC, scale to clock
    OCR1A = 1;

    pinMode(LORA_RST, OUTPUT);
    digitalWrite(LORA_RST, HIGH);
    delay(100);

    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, HIGH); //Sensor power on
    delay(100);

    Lora_init();

    Wire.begin();
    if (humiditySensor.begin() == false)
    {

#if DEBUG_OUT_ENABLE
        Serial.println("AHT10 not detected. Please check wiring. Freezing.");
#endif
    }
#if DEBUG_OUT_ENABLE
    else
        Serial.println("AHT10 acknowledged.");
#endif

    do_some_work();
//setup over
#if DEBUG_OUT_ENABLE
    Serial.println("[Set]Sleep Mode Set");
#endif
    low_power_set();
}

void loop()
{
    wdt_disable();

    if (count > SLEEP_CYCLE) //enough 8 s wake ups have passed, see TRANSMIT_INTERVAL_SECONDS
    {
#if DEBUG_OUT_ENABLE
        //code start
        Serial.println("Code start>>");
#endif

        do_some_work();
        all_pins_low();

#if DEBUG_OUT_ENABLE
        //code end
        Serial.println("Code end<<");
#endif
        //count init
        count = 0;
    }

    low_power_set();
}

ISR(WDT_vect)
{
#if DEBUG_OUT_ENABLE
    Serial.print("[Watch dog]");
    Serial.println(count);
#endif
    delay(100);
    count++;
    //wdt_reset();
    wdt_disable(); // disable watchdog
}

//Set low power mode and into sleep
void low_power_set()
{
    all_pins_low();
    delay(10);
    // disable ADC
    ADCSRA = 0;

    sleep_enable();
    watchdog_init();
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    delay(10);
    noInterrupts();
    sleep_enable();

    // turn off brown-out enable in software
    MCUCR = bit(BODS) | bit(BODSE);
    MCUCR = bit(BODS);
    interrupts();

    sleep_cpu();
    sleep_disable();
}

//Enable watch dog
void watchdog_init()
{
    // clear various "reset" flags
    MCUSR = 0;
    // allow changes, disable reset
    WDTCSR = bit(WDCE) | bit(WDE);
    WDTCSR = bit(WDIE) | bit(WDP3) | bit(WDP0); // set WDIE, and 8 seconds delay
    wdt_reset();                                // pat the dog
}

void do_some_work()
{

    digitalWrite(SENSOR_POWER_PIN, HIGH); // Sensor/RF95 power on
    digitalWrite(LORA_RST, HIGH);
    delay(5);
    pinMode(PWM_OUT_PIN, OUTPUT);    //digitalWrite(PWM_OUT_PIN, LOW);
    TCCR1A = bit(COM1A0);            // toggle OC1A on Compare Match
    TCCR1B = bit(WGM12) | bit(CS10); // CTC, scale to clock
    OCR1A = 1;                       // compare A register value (5000 * clock speed / 1024).When OCR1A == 1, PWM is 2MHz

    Lora_init();
    delay(50);

    //ADC2  AVCC as reference voltage
    ADMUX = _BV(REFS0) | _BV(MUX1);

    //ADC2  internal 1.1V as ADC reference voltage
    //ADMUX = _BV(REFS1) |_BV(REFS0) | _BV(MUX1);

    // enable the ADC with a clock prescaler of 8 (ADPS1 | ADPS0)
    ADCSRA = _BV(ADEN) | _BV(ADPS1) | _BV(ADPS0);
    delay(50);
    for (int i = 0; i < 3; i++)
    {
        //start ADC conversion
        ADCSRA |= (1 << ADSC);

        delay(10);

        if ((ADCSRA & 0x40) == 0)
        {
            ADC_O_1 = ADCL;
            ADC_O_2 = ADCH;

            sensorValue = (ADC_O_2 << 8) + ADC_O_1;
            ADCSRA |= 0x40;
#if DEBUG_OUT_ENABLE
            Serial.print("ADC:");
            Serial.println(sensorValue);
#endif

            if (readSensorStatus == false)
                readSensorStatus = AHT_init();
        }
        ADCSRA |= (1 << ADIF); //reset as required
        delay(50);
    }

    //ADC3  internal 1.1V as ADC reference voltage
    ADMUX = _BV(REFS1) | _BV(REFS0) | _BV(MUX1) | _BV(MUX0);

    delay(50);
    for (int i = 0; i < 3; i++)
    {
        //start ADC conversion
        ADCSRA |= (1 << ADSC);

        delay(10);
        
        if ((ADCSRA & 0x40) == 0)
        {
            ADC_O_1 = ADCL;
            ADC_O_2 = ADCH;

            batValue = (ADC_O_2 << 8) + ADC_O_1;
            ADCSRA |= 0x40;
#if DEBUG_OUT_ENABLE
            Serial.print("BAT:");
            Serial.println(batValue);
            float bat = (float)batValue * 3.3;
            bat = bat / 1024.0;
            Serial.print(bat);
            Serial.println("V");
#endif
        }
        ADCSRA |= (1 << ADIF); //reset as required
        delay(50);
    }
    send_lora();
    delay(1000);
    radio.sleep();

    packetnum++;
    readSensorStatus = false;
    digitalWrite(SENSOR_POWER_PIN, LOW); // Sensor/RF95 power off
    delay(100);
}

void all_pins_low()
{
    pinMode(PWM_OUT_PIN, INPUT);
    pinMode(A4, INPUT_PULLUP);
    pinMode(A5, INPUT_PULLUP);

    delay(50);
}

// Payload format: ID<node> INDEX:<n> H:<h> T:<t> ADC:<adc> BAT:<bat>
// subscriber.py parses these keys with SENSOR_READINGS_FIELDS, so both must change together
void send_lora()
{
    String message = " INDEX:" + (String)packetnum + " H:" + (String)humidity + " T:" + (String)temperature + " ADC:" + (String)sensorValue + " BAT:" + (String)batValue;
    String back_str = node_id + message;

#if DEBUG_OUT_ENABLE
    Serial.println(back_str);
#endif

    radio.transmit(back_str);
}