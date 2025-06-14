#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <ArduinoBLE.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
MAX30105 particleSensor;

const int BUTTON_PIN_1 = 4;
const int BUTTON_PIN_2 = 5;
const int BUTTON_PIN_3 = 6;
const int BUZZER_PIN = 3;

long totalBPM = 0;
int countBPM = 0;
long measurementStartTime = 0;
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;
float spo2Value = 0;
int xPos = 0;
int lastValue = SCREEN_HEIGHT / 2;
int currentMode = 0;

// BLE definitions
BLEService healthService("180D");
BLEFloatCharacteristic bpmCharacteristic("2A37", BLERead | BLENotify);
BLEFloatCharacteristic spo2Characteristic("2A5F", BLERead | BLENotify);
BLEIntCharacteristic ecgCharacteristic("2A58", BLERead | BLENotify);

void setup() {
  Serial.begin(115200);
  
  pinMode(BUTTON_PIN_1, INPUT_PULLUP);
  pinMode(BUTTON_PIN_2, INPUT_PULLUP);
  pinMode(BUTTON_PIN_3, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);

  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("3hmed_3li");
  healthService.addCharacteristic(bpmCharacteristic);
  healthService.addCharacteristic(spo2Characteristic);
  healthService.addCharacteristic(ecgCharacteristic);
  BLE.addService(healthService);
  BLE.advertise();

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED init failed"));
    while (true);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("Initializing...");
  display.display();
  delay(2000);

  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found");
    while (true);
  }

  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);

  measurementStartTime = millis();
}

void loop() {
  BLE.poll();
  checkButtonPress();

  if (currentMode == 0) {
    measureHeartRate();
  } else if (currentMode == 1) {
    measureSpO2();
  } else if (currentMode == 2) {
    measureECGWithAnalysis();  // ✅ بدون فحص leads off
  }
}

void checkButtonPress() {
  static unsigned long lastDebounceTime = 0;
  static int lastButtonState1 = HIGH;
  static int lastButtonState2 = HIGH;
  static int lastButtonState3 = HIGH;

  int reading1 = digitalRead(BUTTON_PIN_1);
  int reading2 = digitalRead(BUTTON_PIN_2);
  int reading3 = digitalRead(BUTTON_PIN_3);

  if (reading1 != lastButtonState1 || reading2 != lastButtonState2 || reading3 != lastButtonState3) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > 50) {
    if (reading1 == LOW) currentMode = 0;
    if (reading2 == LOW) currentMode = 1;
    if (reading3 == LOW) {
      currentMode = 2;
      xPos = 0;
      lastValue = SCREEN_HEIGHT / 2;
      display.clearDisplay();  // ⬅️ امسح الشاشة عند بدء ECG
    }
  }

  lastButtonState1 = reading1;
  lastButtonState2 = reading2;
  lastButtonState3 = reading3;
}

bool isFingerPlaced = false;  // متغير لمتابعة حالة الإصبع (إذا كان الموضوع موجودًا)
unsigned long lastAverageTime = 0;  // لحفظ آخر وقت تحديث المعدل
int countdownTime = 60;  // العد التنازلي للدقيقة
unsigned long lastClearTime = 0;  // لحفظ آخر وقت مسح الشاشة

void measureHeartRate() {
  long irValue = particleSensor.getIR();

  if (irValue > 7000) {  // إذا كانت الإشارة قوية
    if (!isFingerPlaced) {
      isFingerPlaced = true;
      // إعادة تعيين القياسات والعد التنازلي
      totalBPM = 0;
      countBPM = 0;
      measurementStartTime = millis();  // إعادة بدء العد التنازلي
      countdownTime = 60;  // إعادة تعيين العد التنازلي إلى 60 ثانية
    }

    if (checkForBeat(irValue)) {  // اكتشاف نبضة قلب جديدة
      long delta = millis() - lastBeat;
      lastBeat = millis();
      beatsPerMinute = 60 / (delta / 1000.0);

      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;

        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++)
          beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
        beatAvg += 30;

        totalBPM += beatAvg;
        countBPM++;

        tone(BUZZER_PIN, 1000, 100);  // إصدار الصوت عند اكتشاف نبضة قلب جديدة
      }
    }

    // تحديث العد التنازلي
    if (millis() - lastAverageTime >= 1000) {
      countdownTime--;
      lastAverageTime = millis();
    }

    // عرض الوقت المتبقي على الشاشة
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 5);
    display.print("Time Left: ");
    display.print(countdownTime);
    display.println(" sec");

    // عرض معدل ضربات القلب
    display.setTextSize(2);
    display.setCursor(0, 30);
    display.print("BPM: ");
    display.print(beatAvg);
    display.display();

    // حساب وعرض المعدل المتوسط
    if (countdownTime <= 0) {  // بعد انتهاء العد التنازلي
      int averageBPM = totalBPM / countBPM;

      // طباعة المعدل المتوسط في الـ Serial Monitor
      Serial.print("Average BPM (1 min): ");
      Serial.println(averageBPM);

      // عرض المعدل المتوسط على الشاشة
      display.clearDisplay();
      display.setTextSize(2);
      display.setCursor(0, 0);
      display.println("Avg BPM:");
      display.setCursor(10, 30);
      display.println(averageBPM);
      display.display();
      display.clearDisplay();

      tone(BUZZER_PIN, 1000, 1000);  // إصدار الصوت لمدة 1 ثانية عند نهاية الدقيقة
      delay(1000);
      noTone(BUZZER_PIN);  // إيقاف الصوت
      delay(3000);  // الانتظار 3 ثوانٍ قبل مسح الشاشة
      display.clearDisplay();

      // مسح الشاشة بعد 5 ثوانٍ من عرض النتيجة
      if (millis() - lastClearTime >= 5000) {
        display.clearDisplay();
        display.display();
        lastClearTime = millis();  // تحديث آخر وقت مسح الشاشة

        // إعادة تعيين العد التنازلي
        totalBPM = 0;
        countBPM = 0;
        countdownTime = 60;  // إعادة تعيين العد التنازلي إلى 60 ثانية
      }
    }
  } else {
    // إذا كانت الإشارة ضعيفة (الإصبع غير موجود)
    if (isFingerPlaced) {
      display.clearDisplay();
      isFingerPlaced = false;  // تم رفع الإصبع
      display.clearDisplay();
      display.setTextSize(1);
      display.setCursor(10, 20);
      display.println("Place Finger BPM");
      display.display();
      noTone(BUZZER_PIN);  // إيقاف الصوت إذا لم يكن هناك نبضة قلب
    }
  }
}

void measureSpO2() {
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();

  display.clearDisplay();
  display.setTextSize(1);

  if (irValue > 7000) {
    float ratio = (float)redValue / (float)irValue;
    spo2Value = 104 - (17 * ratio);

    spo2Characteristic.writeValue(spo2Value);  // Send to BLE

    Serial.print("SpO2: ");
    Serial.print(spo2Value);
    Serial.println(" %");

    display.setCursor(0, 0);
    display.println("SpO2 Measuring...");
    display.setTextSize(2);
    display.setCursor(10, 30);
    display.print(spo2Value);
    display.println("%");
  } else {
    display.setCursor(10, 20);
    display.println("Place Finger SpO2");
  }
  display.display();
  delay(1000);
}

int getEcgStatus(int ecgValue) {
  const int LOWER_BOUND = 200;
  const int UPPER_BOUND = 800;
  return (ecgValue >= LOWER_BOUND && ecgValue <= UPPER_BOUND) ? 1 : 0;
}

#define ECG_BUFFER_SIZE SCREEN_WIDTH
int ecgBuffer[ECG_BUFFER_SIZE];

void measureECGWithAnalysis() {
  static unsigned long ecgStartTime = 0;
  static int normalReadings = 0;
  static int totalReadings = 0;

  int ecgValue = analogRead(A0);
  Serial.println(ecgValue);

  totalReadings++;
  if (ecgValue >= 200 && ecgValue <= 800) {
    normalReadings++;
  }

  int yValue = map(ecgValue, 0, 1023, SCREEN_HEIGHT - 1, 0);

  // حرك القيم في البفر لليسار
  for (int i = 0; i < ECG_BUFFER_SIZE - 1; i++) {
    ecgBuffer[i] = ecgBuffer[i + 1];
  }
  ecgBuffer[ECG_BUFFER_SIZE - 1] = yValue;

  // رسم الإشارة
  display.clearDisplay();
  for (int x = 1; x < ECG_BUFFER_SIZE; x++) {
    display.drawLine(x - 1, ecgBuffer[x - 1], x, ecgBuffer[x], WHITE);
  }
  display.display();
  delay(10);

  if (ecgStartTime == 0) ecgStartTime = millis();

  if (millis() - ecgStartTime >= 30000) {
    int finalStatus = (normalReadings >= (totalReadings / 2)) ? 0 : 1;

    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println(finalStatus == 0 ? "Status:Normal" : "Status:Abnormal");
    display.display();

    Serial.print(" after 20s: ");
    Serial.println(finalStatus);

    delay(5000); // عرض النتيجة 5 ثوانٍ
    display.clearDisplay();
    ecgStartTime = 0;
    normalReadings = 0;
    totalReadings = 0;

    // تصفير البفر
    for (int i = 0; i < ECG_BUFFER_SIZE; i++) {
      ecgBuffer[i] = SCREEN_HEIGHT / 2;
    }
  }
}
