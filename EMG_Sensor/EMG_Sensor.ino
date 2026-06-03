/*
* Copyright 2017, OYMotion Inc.
* All rights reserved.
*/

#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#include "EMGFilters.h"

#define TIMING_DEBUG 1

#define SensorInputPin_Inside A0   // 팔뚝 안쪽 센서
#define SensorInputPin_Outside A1  // 팔뚝 바깥쪽 센서

EMGFilters filterInside;
EMGFilters filterOutside;

// discrete filters must works with fixed sample frequence
// our emg filter only support "SAMPLE_FREQ_500HZ" or "SAMPLE_FREQ_1000HZ"
int sampleRate = SAMPLE_FREQ_1000HZ;

// For countries where power transmission is at 50 Hz
// For countries where power transmission is at 60 Hz, need to change to
// "NOTCH_FREQ_60HZ"
int humFreq = NOTCH_FREQ_60HZ;

// Calibration:
// put on the sensors, and release your muscles;
// wait a few seconds, and select the max value as the threshold;
// any value under threshold will be set to zero
static long ThresholdInside = 0;
static long ThresholdOutside = 0;

unsigned long timeStamp;
unsigned long timeBudget;

void setup() {
    filterInside.init(sampleRate, humFreq, true, true, true);
    filterOutside.init(sampleRate, humFreq, true, true, true);

    Serial.begin(115200);

    timeBudget = 1000000 / sampleRate;
}

void loop() {
    timeStamp = micros();

    int valueInside = analogRead(SensorInputPin_Inside);
    int valueOutside = analogRead(SensorInputPin_Outside);

    // filter processing
    int dataInsideAfterFilter = filterInside.update(valueInside);
    int dataOutsideAfterFilter = filterOutside.update(valueOutside);

    long envelopeInside = sq((long)dataInsideAfterFilter);
    long envelopeOutside = sq((long)dataOutsideAfterFilter);

    // any value under threshold will be set to zero
    envelopeInside = (envelopeInside > ThresholdInside) ? envelopeInside : 0;
    envelopeOutside = (envelopeOutside > ThresholdOutside) ? envelopeOutside : 0;

    if (TIMING_DEBUG) {
      
        // Serial Plotter용 출력
        // 안쪽 센서, 바깥쪽 센서 순서
        Serial.print(envelopeInside);
        Serial.print(",");
        Serial.println(envelopeOutside);
    }
    // 1000Hz 주기 맞추기
    while (micros() - timeStamp < timeBudget) {
    // wait
    }
}
