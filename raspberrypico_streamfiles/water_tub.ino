 
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* mqtt_server = "test.mosquitto.org";
const int mqtt_port = 1883;
int sensorPin = 26;
float volt;
float ntu;
float tempC;

const char* ssid = "Rono";
const char* password = "88888888";

WiFiClient espClient;
PubSubClient client(espClient);

long randNumber;
StaticJsonDocument<256> doc;

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "mqtt-explorer-584bcd18";
    clientId += String(random(0xffff), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("Connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {

  // Start the Serial Monitor
  Serial.begin(115200);
  analogReadResolution(10);
  delay(1000);
  WiFi.mode(WIFI_STA);

  // Start WiFi with supplied parameters
  WiFi.begin(ssid, password);

  // Print periods on monitor while establishing connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    delay(500);
  }
  client.setServer(mqtt_server, mqtt_port);
  delay(100);


}
 
void loop()
{
    tempC = analogReadTemp();
    int r=analogRead(sensorPin);
    float kenvals=map(r,0,1023,0,320);

    volt = 0;
    for(int i=0; i<800; i++)
    {
        volt += ((float)analogRead(sensorPin)/4095)*3.3;
    }
    volt = volt/800;
    volt = round_to_dp(volt,2);
    if(volt < 0){
      ntu = 3000;
    }else{
      ntu = -1120.4*(volt*volt)+5742.3*volt-4353.8; 
    }
    delay(1);
    if (!client.connected()) {
    reconnect();
  }
  client.loop();
    String water_level =String(kenvals);
    String flow_rate=String(tempC);

    JsonArray datajson = doc.createNestedArray("datajson");
    datajson.add(water_level);
    datajson.add(flow_rate);
    char out[128];
    int b =serializeJson(doc, out);
    boolean dataout= client.publish("ken/data", out);
    
    Serial.println("Data published");
    delay(1000);
    Serial.println(ntu);
    Serial.println(kenvals);
    Serial.println(tempC);
}
 
float round_to_dp( float in_value, int decimal_place )
{
  float multiplier = powf( 10.0f, decimal_place );
  in_value = roundf( in_value * multiplier ) / multiplier;
  return in_value;
}