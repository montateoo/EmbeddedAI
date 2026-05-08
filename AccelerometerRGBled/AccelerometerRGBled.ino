#include <Arduino_LSM9DS1.h>
#define LEDR 22
#define LEDG 23
#define LEDB 24


void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Acceleration in g's");
  Serial.println("X\tY\tZ");

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
}

void loop() {
  float x, y, z;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    Serial.print(x);
    Serial.print('\t');
    Serial.print(y);
    Serial.print('\t');
    Serial.println(z);

    /*
    if(x>0){
      digitalWrite(LEDR,LOW);  
    } else {
      digitalWrite(LEDR,HIGH);
    }

    if(y>0){
      digitalWrite(LEDG,LOW);  
    } else {
      digitalWrite(LEDG,HIGH);
    }

    if(z>0){
      digitalWrite(LEDB,LOW);  
    } else {
      digitalWrite(LEDB,HIGH);
    }
    */

    if(x>y && x>z){
      digitalWrite(LEDR,LOW);
      digitalWrite(LEDG,HIGH);
      digitalWrite(LEDB,HIGH);
    }
    else if(y>z && y>x){
      digitalWrite(LEDR,HIGH);
      digitalWrite(LEDG,LOW);
      digitalWrite(LEDB,HIGH);
    }
    else{
      digitalWrite(LEDR,HIGH);
      digitalWrite(LEDG,HIGH);
      digitalWrite(LEDB,LOW);
    }
        
  }
}
