/*
 * EMG 5-class classifier serial output
 *
 * Labels:
 *   0 = rock
 *   1 = paper
 *   2 = none
 *   3 = middle
 *   4 = thumb
 *
 * Feature order:
 *   x[0] = MAV_ch1
 *   x[1] = MAV_ch2
 *   x[2] = RMS_ch1
 *   x[3] = RMS_ch2
 *   x[4] = WL_ch1
 *   x[5] = WL_ch2
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
#define ServoPin1 2
#define ServoPin2 3
#define ServoPin3 4

const int SAMPLE_RATE = SAMPLE_FREQ_1000HZ;
const int HUM_FREQ = NOTCH_FREQ_60HZ;

const unsigned long SAMPLE_PERIOD_US = 1000000UL / 1000UL;
const int WINDOW_SIZE = 200;
const int STEP_SIZE = 100;
const int FEATURE_COUNT = 15;
const int VOTE_WINDOW_SIZE = 9;
const int VOTE_MIN_COUNT = 5;
const int LABEL_COUNT = 5;
const float BALANCE_EPSILON = 0.001f;

static long ThresholdInside = 0;
static long ThresholdOutside = 0;

EMGFilters filterInside;
EMGFilters filterOutside;
Servo fingerServo1;
Servo fingerServo2;
Servo fingerServo3;

uint16_t windowInside[WINDOW_SIZE];
uint16_t windowOutside[WINDOW_SIZE];
int windowIndex = 0;
int samplesInWindow = 0;
int samplesSincePredict = 0;
int lastServoLabel = -1;
int lastServoAngle = 90;
int labelHistory[VOTE_WINDOW_SIZE];
int labelHistoryIndex = 0;
int labelsInHistory = 0;
int stableLabel = 2;

int predictEMG(float *x) {
    if (x[11] <= 2081.5f) {
        return 2;
    } else {
        if (x[12] <= 0.006119620055f) {
            if (x[4] <= 19607.5f) {
                if (x[6] <= -0.6599999964f) {
                    if (x[11] <= 8782.0f) {
                        return 0;
                    } else {
                        return 4;
                    }
                } else {
                    if (x[2] <= 67.77761078f) {
                        return 4;
                    } else {
                        return 0;
                    }
                }
            } else {
                if (x[12] <= -0.01477864059f) {
                    if (x[1] <= 123.8025017f) {
                        return 4;
                    } else {
                        return 0;
                    }
                } else {
                    return 0;
                }
            }
        } else {
            if (x[6] <= 7.277499914f) {
                if (x[2] <= 80.17727661f) {
                    if (x[1] <= 32.24500084f) {
                        return 1;
                    } else {
                        return 1;
                    }
                } else {
                    if (x[12] <= 0.01409404585f) {
                        return 3;
                    } else {
                        return 3;
                    }
                }
            } else {
                return 1;
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
        case 3:
            return "middle";
        case 4:
            return "thumb";
        default:
            return "unknown";
    }
}

void servoAnglesForLabel(int label, int &angle1, int &angle2, int &angle3) {
    switch (label) {
        case 0:
            angle1 = 180;
            angle2 = 0;
            angle3 = 180;
            break;
        case 1:
            angle1 = 0;
            angle2 = 180;
            angle3 = 0;
            break;
        case 2:
            angle1 = 90;
            angle2 = 90;
            angle3 = 90;
            break;
        case 3:  // middle
            angle1 = 0;
            angle2 = 180;
            angle3 = 0;
            break;
        case 4:  // thumb
            angle1 = 0;
            angle2 = 0;
            angle3 = 180;
            break;
        default:
            angle1 = 90;
            angle2 = 90;
            angle3 = 90;
            break;
    }
}

void updateServo(int label) {
    if (label == lastServoLabel) {
        return;
    }

    int angle1;
    int angle2;
    int angle3;
    servoAnglesForLabel(label, angle1, angle2, angle3);
    fingerServo1.write(angle1);
    fingerServo2.write(angle2);
    fingerServo3.write(angle3);
    lastServoLabel = label;
}

int smoothLabel(int label) {
    labelHistory[labelHistoryIndex] = label;
    labelHistoryIndex = (labelHistoryIndex + 1) % VOTE_WINDOW_SIZE;

    if (labelsInHistory < VOTE_WINDOW_SIZE) {
        labelsInHistory++;
    }

    int counts[LABEL_COUNT] = {0};
    for (int i = 0; i < labelsInHistory; i++) {
        int historyLabel = labelHistory[i];
        if (historyLabel >= 0 && historyLabel < LABEL_COUNT) {
            counts[historyLabel]++;
        }
    }

    int bestLabel = stableLabel;
    int bestCount = counts[stableLabel];
    for (int candidate = 0; candidate < LABEL_COUNT; candidate++) {
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

    features[0] = sumInside / WINDOW_SIZE;
    features[1] = sumOutside / WINDOW_SIZE;
    features[2] = sqrt(squareSumInside / WINDOW_SIZE);
    features[3] = sqrt(squareSumOutside / WINDOW_SIZE);
    features[4] = wlInside;
    features[5] = wlOutside;

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
    fingerServo1.attach(ServoPin1);
    fingerServo2.attach(ServoPin2);
    fingerServo3.attach(ServoPin3);
    fingerServo1.write(90);
    fingerServo2.write(90);
    fingerServo3.write(90);

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
