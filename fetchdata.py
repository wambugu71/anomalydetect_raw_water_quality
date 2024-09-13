import paho.mqtt.client as mqtt
import time


# Broker connection
mqtt_broker = "test.mosquitto.org"
mqtt_port = 1883
mqtt_topic = "ken/data"

in_val = None

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe(mqtt_topic)

def on_message(client, userdata, msg):
    global info_text
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
    
    

def mqtt_function(mqtt_broker,mqtt_port,mqtt_topic):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(mqtt_broker, mqtt_port, 60)
    client.loop_forever()

while True:
    mqtt_function(mqtt_broker,mqtt_port,mqtt_topic)
    time.sleep(1)
