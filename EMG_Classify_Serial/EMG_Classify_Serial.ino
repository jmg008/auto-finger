/*
 * EMG 3-class classifier serial output
 *
 * Labels:
 *   0 = rock
 *   1 = paper
 *   2 = none
 *
 * Feature order:
 *   x[0] = MAV_ch1_norm
 *   x[1] = MAV_ch2_norm
 *   x[2] = RMS_ch1_norm
 *   x[3] = RMS_ch2_norm
 *   x[4] = WL_ch1_norm
 *   x[5] = WL_ch2_norm
 *   x[6] = MAV_diff
 *   x[7] = RMS_diff
 *   x[8] = WL_diff
 *   x[9] = MAV_sum
 *   x[10] = RMS_sum
 *   x[11] = WL_sum
 *   x[12] = MAV_balance
 *   x[13] = RMS_balance
 *   x[14] = WL_balance
 */

#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#include "EMGFilters.h"
#include <Servo.h>

#define SensorInputPin_Inside A0
#define SensorInputPin_Outside A1
#define ServoPin 2

const int SAMPLE_RATE = SAMPLE_FREQ_1000HZ;
const int HUM_FREQ = NOTCH_FREQ_60HZ;

const unsigned long SAMPLE_PERIOD_US = 1000000UL / 1000UL;
const int WINDOW_SIZE = 200;
const int STEP_SIZE = 50;
const int RAW_FEATURE_COUNT = 6;
const int FEATURE_COUNT = 15;
const int VOTE_WINDOW_SIZE = 9;
const int VOTE_MIN_COUNT = 5;
const float NORMALIZATION_MIN_DENOM = 0.001f;
const float BALANCE_EPSILON = 0.001f;
const float NORMALIZATION_REST[RAW_FEATURE_COUNT] = { 21.405f, 21.425f, 27.26554602f, 27.2797544f, 2791.0f, 2786.0f };
const float NORMALIZATION_CALIB[RAW_FEATURE_COUNT] = { 597.597f, 585.375f, 1191.212029f, 1190.935248f, 112342.0f, 111811.75f };

static long ThresholdInside = 0;
static long ThresholdOutside = 0;

EMGFilters filterInside;
EMGFilters filterOutside;
Servo fingerServo;

uint16_t windowInside[WINDOW_SIZE];
uint16_t windowOutside[WINDOW_SIZE];
int windowIndex = 0;
int samplesInWindow = 0;
int samplesSincePredict = 0;
int lastServoAngle = -1;
int labelHistory[VOTE_WINDOW_SIZE];
int labelHistoryIndex = 0;
int labelsInHistory = 0;
int stableLabel = 2;

int predictEMG(float *x) {
    if (x[11] <= 0.179981017f) {
        return 2;
    } else {
        if (x[12] <= -0.01621893141f) {
            return 1;
        } else {
            if (x[8] <= -0.04715042002f) {
                if (x[3] <= 0.7052450478f) {
                    return 1;
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        }
    }
}

const char *labelName(int label) {
    switch (label) {
        case 0:
            return "rock";
        case 1:
            return "paper";
        case 2:
            return "none";
        default:
            return "unknown";
    }
}

int servoAngleForLabel(int label) {
    switch (label) {
        case 0:
            return 180;
        case 1:
            return 0;
        case 2:
            return 90;
        default:
            return 90;
    }
}

void updateServo(int label) {
    int angle = servoAngleForLabel(label);

    if (angle != lastServoAngle) {
        fingerServo.write(angle);
        lastServoAngle = angle;
    }
}

int smoothLabel(int label) {
    labelHistory[labelHistoryIndex] = label;
    labelHistoryIndex = (labelHistoryIndex + 1) % VOTE_WINDOW_SIZE;

    if (labelsInHistory < VOTE_WINDOW_SIZE) {
        labelsInHistory++;
    }

    int counts[3] = {0, 0, 0};
    for (int i = 0; i < labelsInHistory; i++) {
        int historyLabel = labelHistory[i];
        if (historyLabel >= 0 && historyLabel <= 2) {
            counts[historyLabel]++;
        }
    }

    int bestLabel = stableLabel;
    int bestCount = counts[stableLabel];
    for (int candidate = 0; candidate < 3; candidate++) {
        if (counts[candidate] > bestCount) {
            bestLabel = candidate;
            bestCount = counts[candidate];
        }
    }

    if (labelsInHistory == VOTE_WINDOW_SIZE && bestCount >= VOTE_MIN_COUNT) {
        stableLabel = bestLabel;
    }

    return stableLabel;
}

float normalizeFeature(float value, int index) {
    float denom = NORMALIZATION_CALIB[index] - NORMALIZATION_REST[index];
    if (denom < NORMALIZATION_MIN_DENOM) {
        denom = NORMALIZATION_MIN_DENOM;
    }

    float normalized = (value - NORMALIZATION_REST[index]) / denom;
    if (normalized < 0.0f) {
        return 0.0f;
    }

    return normalized;
}

uint16_t clampEnvelope(long envelope) {
    if (envelope < 0) {
        return 0;
    }

    if (envelope > 65535) {
        return 65535;
    }

    return (uint16_t)envelope;
}

float getInsideAt(int offset) {
    int index = (windowIndex + offset) % WINDOW_SIZE;
    return windowInside[index];
}

float getOutsideAt(int offset) {
    int index = (windowIndex + offset) % WINDOW_SIZE;
    return windowOutside[index];
}

void calculateFeatures(float *features) {
    float sumInside = 0.0;
    float sumOutside = 0.0;
    float squareSumInside = 0.0;
    float squareSumOutside = 0.0;
    float wlInside = 0.0;
    float wlOutside = 0.0;

    float prevInside = getInsideAt(0);
    float prevOutside = getOutsideAt(0);

    for (int i = 0; i < WINDOW_SIZE; i++) {
        float inside = getInsideAt(i);
        float outside = getOutsideAt(i);

        sumInside += abs(inside);
        sumOutside += abs(outside);
        squareSumInside += inside * inside;
        squareSumOutside += outside * outside;

        if (i > 0) {
            wlInside += abs(inside - prevInside);
            wlOutside += abs(outside - prevOutside);
        }

        prevInside = inside;
        prevOutside = outside;
    }

    float rawFeatures[RAW_FEATURE_COUNT];
    rawFeatures[0] = sumInside / WINDOW_SIZE;
    rawFeatures[1] = sumOutside / WINDOW_SIZE;
    rawFeatures[2] = sqrt(squareSumInside / WINDOW_SIZE);
    rawFeatures[3] = sqrt(squareSumOutside / WINDOW_SIZE);
    rawFeatures[4] = wlInside;
    rawFeatures[5] = wlOutside;

    for (int i = 0; i < RAW_FEATURE_COUNT; i++) {
        features[i] = normalizeFeature(rawFeatures[i], i);
    }

    features[6] = features[0] - features[1];
    features[7] = features[2] - features[3];
    features[8] = features[4] - features[5];
    features[9] = features[0] + features[1];
    features[10] = features[2] + features[3];
    features[11] = features[4] + features[5];
    features[12] = features[6] / (features[9] + BALANCE_EPSILON);
    features[13] = features[7] / (features[10] + BALANCE_EPSILON);
    features[14] = features[8] / (features[11] + BALANCE_EPSILON);
}

void setup() {
    filterInside.init(SAMPLE_RATE, HUM_FREQ, true, true, true);
    filterOutside.init(SAMPLE_RATE, HUM_FREQ, true, true, true);
    fingerServo.attach(ServoPin);
    fingerServo.write(90);
    lastServoAngle = 90;

    Serial.begin(115200);
}

void loop() {
    unsigned long timeStamp = micros();

    int valueInside = analogRead(SensorInputPin_Inside);
    int valueOutside = analogRead(SensorInputPin_Outside);

    int dataInsideAfterFilter = filterInside.update(valueInside);
    int dataOutsideAfterFilter = filterOutside.update(valueOutside);

    long envelopeInside = sq((long)dataInsideAfterFilter);
    long envelopeOutside = sq((long)dataOutsideAfterFilter);

    envelopeInside = (envelopeInside > ThresholdInside) ? envelopeInside : 0;
    envelopeOutside = (envelopeOutside > ThresholdOutside) ? envelopeOutside : 0;

    windowInside[windowIndex] = clampEnvelope(envelopeInside);
    windowOutside[windowIndex] = clampEnvelope(envelopeOutside);
    windowIndex = (windowIndex + 1) % WINDOW_SIZE;

    if (samplesInWindow < WINDOW_SIZE) {
        samplesInWindow++;
    }

    samplesSincePredict++;

    if (samplesInWindow == WINDOW_SIZE && samplesSincePredict >= STEP_SIZE) {
        float features[FEATURE_COUNT];
        calculateFeatures(features);

        int label = smoothLabel(predictEMG(features));
        updateServo(label);

        Serial.print(label);
        Serial.print(",");
        Serial.println(labelName(label));

        samplesSincePredict = 0;
    }

    while (micros() - timeStamp < SAMPLE_PERIOD_US) {
    }
}
