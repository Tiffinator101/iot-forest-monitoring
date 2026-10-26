#if !defined(_RADIOLIB_USER_BUILD_OPTIONS_H)
#define _RADIOLIB_USER_BUILD_OPTIONS_H

// this file can be used to define any user build options
// most commonly, RADIOLIB_EXCLUDE_* macros
// or enabling debug output

//#define RADIOLIB_DEBUG    (1)
//#define RADIOLIB_VERBOSE  (1)

// the sensor node only uses the SX1276 (SX127X driver) to send LoRa packets,
// so every other driver and protocol is excluded to fit the ATmega328P's 30 KB of flash

// other radio drivers
#define RADIOLIB_EXCLUDE_CC1101           (1)
#define RADIOLIB_EXCLUDE_NRF24            (1)
#define RADIOLIB_EXCLUDE_RF69             (1)
#define RADIOLIB_EXCLUDE_SX1231           (1)
#define RADIOLIB_EXCLUDE_SI443X           (1)
#define RADIOLIB_EXCLUDE_RFM2X            (1)
#define RADIOLIB_EXCLUDE_SX126X           (1)
#define RADIOLIB_EXCLUDE_STM32WLX         (1)
#define RADIOLIB_EXCLUDE_SX128X           (1)

// protocols not used by the sensor node
#define RADIOLIB_EXCLUDE_AFSK             (1)
#define RADIOLIB_EXCLUDE_AX25             (1)
#define RADIOLIB_EXCLUDE_APRS             (1)
#define RADIOLIB_EXCLUDE_BELL             (1)
#define RADIOLIB_EXCLUDE_FSK4             (1)
#define RADIOLIB_EXCLUDE_HELLSCHREIBER    (1)
#define RADIOLIB_EXCLUDE_MORSE            (1)
#define RADIOLIB_EXCLUDE_PAGER            (1)
#define RADIOLIB_EXCLUDE_RTTY             (1)
#define RADIOLIB_EXCLUDE_SSTV             (1)
#define RADIOLIB_EXCLUDE_LORAWAN          (1)

// direct mode receive, only used by non-LoRa protocols
#define RADIOLIB_EXCLUDE_DIRECT_RECEIVE   (1)

#endif
