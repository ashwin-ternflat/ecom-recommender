import json, time, random
from datetime import datetime
from kafka import KafkaProducer

user_ids = [f"user_{i}" for i in range(1, 11)]
item_ids = [f"item_{i}" for i in range(1, 21)]
actions = ["click", "view", "add_to_cart", "purchase"]

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_event():
    return {
        "user_id": random.choice(user_ids),
        "item_id": random.choice(item_ids),
        "action": random.choice(actions),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == '__main__':
    topic = 'user_events'
    while True:
        event = generate_event()
        print(f"Sending event: {event}")
        producer.send(topic, value=event)
        time.sleep(1)
