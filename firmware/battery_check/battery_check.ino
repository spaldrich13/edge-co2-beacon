// battery_check.ino
// Reads MAX17048 fuel gauge via I2C and prints SoC + voltage every 2s.
// Library: Adafruit MAX1704X  (install via Arduino Library Manager)
// Board:   Adafruit Feather nRF52840

#include <Wire.h>
#include <Adafruit_MAX1704X.h>   // Library Manager: "Adafruit MAX1704X"

Adafruit_MAX17048 maxlipo;

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("MAX17048 battery check");

    Wire.begin();
    Serial.println("Scanning I2C bus...");
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.print("  Found device at 0x");
            Serial.println(addr, HEX);
        }
    }
    Serial.println("Scan complete.");

    if (!maxlipo.begin()) {
        Serial.println("ERROR: MAX17048 not found. Check wiring / I2C address 0x36.");
        while (1) delay(1000);
    }

    Serial.println("MAX17048 found OK");
}

void loop() {
    float pct = maxlipo.cellPercent();
    float v   = maxlipo.cellVoltage();

    Serial.print("Battery: ");
    Serial.print(pct, 1);
    Serial.print("%  Voltage: ");
    Serial.print(v, 2);
    Serial.println("V");

    if (pct >= 99.0f) {
        Serial.println("FULLY CHARGED - ready for test");
    }

    delay(2000);
}
