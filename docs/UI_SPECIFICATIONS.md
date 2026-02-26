# UI_SPECIFICATIONS.md — Web Dashboard

## Overview

React single-page app that connects to the beacon via Web Bluetooth and displays
real-time mode classification and trip CO₂ summaries.

**Stack:** React + Web Bluetooth API + React state (no localStorage, no backend)
**Dev command:** `cd dashboard && npm run dev` → Chrome at `http://localhost:3000`
**Browser:** Chrome only (Web Bluetooth not supported in Firefox/Safari)

---

## Layout

```
┌─────────────────────────────────────────────────────────┐
│  CO₂ Beacon Dashboard                     [Connect BLE] │
├──────────────────────┬──────────────────────────────────┤
│  LIVE STATUS         │  TRIP HISTORY                    │
│  ┌────────────────┐  │  ┌──────────────────────────────┐│
│  │ Mode: 🚗 Car   │  │  │ #3  Car  12 min  0.89 km    ││
│  │ Conf: 94%      │  │  │      CO₂: 79 g              ││
│  │ Trip: Active   │  │  ├──────────────────────────────┤│
│  │ 00:04:32       │  │  │ #2  Walk  8 min  0.67 km    ││
│  └────────────────┘  │  │      CO₂: 0 g               ││
│                      │  ├──────────────────────────────┤│
│  CO₂ THIS TRIP       │  │ #1  Train 22 min  29.3 km   ││
│  ┌────────────────┐  │  │      CO₂: 1026 g            ││
│  │    79 g        │  │  └──────────────────────────────┘│
│  └────────────────┘  │                                  │
│                      │  SESSION TOTAL CO₂               │
│  TOTAL SESSION       │  ┌──────────────────────────────┐│
│  ┌────────────────┐  │  │  1105 g  (~avg car: 1.1 km) ││
│  │  1.1 kg CO₂   │  │  └──────────────────────────────┘│
│  └────────────────┘  │                                  │
└──────────────────────┴──────────────────────────────────┘
```

---

## Components

### `<App />`

Root component. Manages BLE connection state and trip data.

**State:**
```js
{
    connected: boolean,
    device: BluetoothDevice | null,
    liveStatus: LiveStatus | null,
    trips: TripRecord[],
    sessionCo2G: number,
}
```

### `<ConnectButton />`

- Label: "Connect BLE" when disconnected, "Disconnect" when connected
- On click (disconnected): calls `navigator.bluetooth.requestDevice(...)` with CO2-Beacon filter
- On click (connected): calls `device.gatt.disconnect()`
- Shows spinner during connection attempt

### `<LiveStatusPanel />`

Displays current mode, confidence, trip timer, and per-trip CO₂.

Props: `{ liveStatus: LiveStatus | null }`

| Field       | Display                                               |
|-------------|-------------------------------------------------------|
| mode_id     | Mode name + emoji (see Mode Display below)            |
| confidence  | "Conf: 94%"                                           |
| trip_active | "Trip: Active" (green) / "Trip: Idle" (gray)          |
| elapsed     | Running timer HH:MM:SS derived from ts_start          |

### `<TripHistoryList />`

Scrollable list of TripRecord entries, newest first.

Props: `{ trips: TripRecord[] }`

Per row:
- `#<trip_id>  <mode_name>  <duration>  <distance_km> km`
- `CO₂: <co2_g> g`

### `<SessionSummary />`

Shows total session CO₂ across all trips in this session.

Props: `{ trips: TripRecord[] }`

Calculation: `sum(trip.co2_g)` → display in g if < 1000, kg if ≥ 1000.

---

## Mode Display

| mode_id | Label  | Emoji |
|---------|--------|-------|
| 0       | Train  | 🚂    |
| 1       | Subway | 🚇    |
| 2       | Car    | 🚗    |
| 3       | Bus    | 🚌    |
| 4       | Walk   | 🚶    |

---

## BLE Data Flow

```
Beacon
  │── Notify LiveStatus (1 Hz) ──► liveStatusChar.addEventListener
  │                                       │
  │                                       ▼
  │                               setLiveStatus(parsed)
  │
  │── Indicate TripRecord ──────► tripRecordChar.addEventListener
                                          │
                                          ▼
                                  setTrips([...trips, parsed])
```

On Control write 0x01 (SYNC_TRIPS): beacon sends all stored TripRecords as
sequential Indicate events on the TripRecord characteristic.

---

## Formatting Helpers

```js
const MODE_NAMES = ['Train', 'Subway', 'Car', 'Bus', 'Walk'];
const MODE_EMOJI = ['🚂', '🚇', '🚗', '🚌', '🚶'];

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m} min ${s} s`;
}

function formatCo2(grams) {
    return grams >= 1000
        ? `${(grams / 1000).toFixed(2)} kg`
        : `${grams} g`;
}

function formatDistance(meters) {
    return meters >= 1000
        ? `${(meters / 1000).toFixed(1)} km`
        : `${meters} m`;
}
```

---

## Constraints

- No localStorage — all state lives in React memory only
- No backend server — purely client-side
- No external API calls
- Web Bluetooth requires HTTPS or localhost
- Must run in Chrome (Web Bluetooth API requirement)
- Dashboard receives only parsed trip summaries — never raw sensor data
