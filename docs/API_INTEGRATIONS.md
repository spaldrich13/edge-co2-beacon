# API_INTEGRATIONS.md — BLE GATT, I²C Setup, CO₂ Factors

## BLE GATT Service

### UUIDs

```c
// Custom 128-bit service UUID
#define CO2_BEACON_SERVICE_UUID  "12345678-1234-1234-1234-123456789000"

// Characteristic UUIDs
#define LIVE_STATUS_CHAR_UUID    "12345678-1234-1234-1234-123456789001"
#define TRIP_RECORD_CHAR_UUID    "12345678-1234-1234-1234-123456789002"
#define CONTROL_CHAR_UUID        "12345678-1234-1234-1234-123456789003"
```

### Characteristic Permissions

| Characteristic | Properties       | CCCD Required |
|----------------|------------------|---------------|
| LiveStatus     | Notify           | Yes           |
| TripRecord     | Read, Indicate   | Yes (Indicate)|
| Control        | Write            | No            |

### Control Commands (1 byte)

| Byte | Command          | Description                        |
|------|------------------|------------------------------------|
| 0x01 | SYNC_TRIPS       | Push all unsent TripRecords        |
| 0x02 | CLEAR_TRIPS      | Wipe trip log on device            |
| 0x03 | SYNC_TIME        | Reserved — future NTP time sync    |

### Arduino BLE Implementation Sketch

```cpp
#include <ArduinoBLE.h>

BLEService co2Service("12345678-1234-1234-1234-123456789000");

BLECharacteristic liveStatusChar(
    "12345678-1234-1234-1234-123456789001",
    BLENotify, 7);

BLECharacteristic tripRecordChar(
    "12345678-1234-1234-1234-123456789002",
    BLERead | BLEIndicate, 20);

BLECharacteristic controlChar(
    "12345678-1234-1234-1234-123456789003",
    BLEWrite, 1);

void setupBLE() {
    BLE.begin();
    BLE.setLocalName("CO2-Beacon");
    BLE.setAdvertisedService(co2Service);

    co2Service.addCharacteristic(liveStatusChar);
    co2Service.addCharacteristic(tripRecordChar);
    co2Service.addCharacteristic(controlChar);

    BLE.addService(co2Service);
    BLE.advertise();
}

// Notify LiveStatus at 1 Hz
void notifyLiveStatus(LiveStatus_t* status) {
    liveStatusChar.writeValue((uint8_t*)status, sizeof(LiveStatus_t));
}

// Indicate a TripRecord
void indicateTripRecord(TripRecord_t* record) {
    tripRecordChar.writeValue((uint8_t*)record, sizeof(TripRecord_t));
}
```

---

## I²C Setup

### Bus Configuration

```cpp
Wire.begin();
Wire.setClock(400000);  // 400 kHz fast mode
```

### BNO055 (IMU) — Address 0x28

```cpp
#include <Adafruit_BNO055.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

void setupBNO055() {
    if (!bno.begin()) {
        // handle error
    }
    delay(1000);
    bno.setExtCrystalUse(true);  // use external crystal for accuracy
}

// Read at 25 Hz
imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
imu::Vector<3> gyro  = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
```

#### Accelerometer Bias (calibrated values)

```cpp
const float ACC_BIAS_X = -0.1926f;  // m/s²
const float ACC_BIAS_Y = -0.1975f;  // m/s²
const float ACC_BIAS_Z = -0.3472f;  // m/s²

float ax_corr = accel.x() - ACC_BIAS_X;
float ay_corr = accel.y() - ACC_BIAS_Y;
float az_corr = accel.z() - ACC_BIAS_Z;
```

### BMP390 (Barometer) — Address 0x76 (hardware), 0x77 (logging firmware)

```cpp
#include <Adafruit_BMP3XX.h>

Adafruit_BMP3XX bmp;

void setupBMP390() {
    if (!bmp.begin_I2C(0x76)) {  // use 0x77 if SDO pin is high
        // handle error
    }
    bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_2X);
    bmp.setPressureOversampling(BMP3_OVERSAMPLING_8X);
    bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
    bmp.setOutputDataRate(BMP3_ODR_50_HZ);
}

// Read pressure
if (bmp.performReading()) {
    float pressure_hpa = bmp.pressure / 100.0f;
}
```

Note: The logging firmware uses address `0x77`; the CLAUDE.md spec says `0x76`.
Confirm which SDO pin configuration is used on the physical hardware before flashing
the inference firmware.

---

## CO₂ Emission Factors

Source: EPA eGRID 2023

```c
static const float CO2_FACTOR_KG_PER_KM[5] = {
    0.035f,  // 0 = train
    0.041f,  // 1 = subway
    0.089f,  // 2 = car
    0.089f,  // 3 = bus
    0.000f,  // 4 = walk
};

float estimate_co2_g(uint8_t mode_id, float distance_km) {
    if (mode_id > 4) return 0.0f;
    return CO2_FACTOR_KG_PER_KM[mode_id] * distance_km * 1000.0f;
}
```

---

## Distance Estimation API

```c
// Average transit speeds (km/h)
static const float AVG_SPEED_KPH[5] = {
    80.0f,   // 0 = train
    30.0f,   // 1 = subway
    40.0f,   // 2 = car
    20.0f,   // 3 = bus
    5.0f,    // 4 = walk
};

float estimate_distance_km(uint8_t mode_id, uint32_t duration_s) {
    if (mode_id > 4) return 0.0f;
    float hours = duration_s / 3600.0f;
    return AVG_SPEED_KPH[mode_id] * hours;
}
```

---

## Web Bluetooth API (Dashboard)

```javascript
// Connect to beacon
async function connectToBeacon() {
    const device = await navigator.bluetooth.requestDevice({
        filters: [{ name: 'CO2-Beacon' }],
        optionalServices: ['12345678-1234-1234-1234-123456789000']
    });
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(
        '12345678-1234-1234-1234-123456789000'
    );
    return service;
}

// Subscribe to LiveStatus notifications
async function subscribeLiveStatus(service, callback) {
    const char = await service.getCharacteristic(
        '12345678-1234-1234-1234-123456789001'
    );
    char.addEventListener('characteristicvaluechanged', (e) => {
        const view = e.target.value;
        callback({
            mode_id:    view.getUint8(0),
            confidence: view.getUint8(1),
            timestamp:  view.getUint32(2, true),  // little-endian
            tripActive: view.getUint8(6) === 1,
        });
    });
    await char.startNotifications();
}

// Parse TripRecord (20 bytes)
function parseTripRecord(dataView) {
    return {
        trip_id:    dataView.getUint16(0, true),
        mode_id:    dataView.getUint8(2),
        confidence: dataView.getUint8(3),
        ts_start:   dataView.getUint32(4, true),
        ts_end:     dataView.getUint32(8, true),
        duration_s: dataView.getUint16(12, true),
        distance_m: dataView.getUint16(14, true),
        co2_g:      dataView.getUint16(16, true),
    };
}
```

---

## TFLite Micro Interpreter Setup

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model_data.h"  // generated from convert_to_tflite.py

// Static arena — no heap
static uint8_t tensor_arena[80 * 1024];

tflite::MicroMutableOpResolver<6> resolver;
resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddReshape();
resolver.AddQuantize();

const tflite::Model* model = tflite::GetModel(g_model_data);
tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, sizeof(tensor_arena));
interpreter.AllocateTensors();

TfLiteTensor* input  = interpreter.input(0);   // shape (1,125,7,1) INT8
TfLiteTensor* output = interpreter.output(0);  // shape (1,5) INT8
```
