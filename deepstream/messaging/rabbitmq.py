import json
import pika
from typing import Dict
import os


class RabbitMQManager:
    def __init__(self, host: str, username: str, password: str):
        self.host = host
        self.credentials = pika.PlainCredentials(username, password)
        self.parameters = pika.ConnectionParameters(host=host, credentials=self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        self.message_ttl = int(os.getenv("MESSAGE_TTL", 10000))
        self.max_length = int(os.getenv("MAX_LENGTH", 1000))

    def create_queue(self, queue: str):
        try:
            self.channel.queue_declare(queue=queue, durable=True, arguments={
                "x-message-ttl": self.message_ttl,
                "x-max-length": self.max_length,
                "x-overflow": "drop-head",
            })
        except pika.exceptions.AMQPError as e:
            raise RuntimeError(f"Failed to create queue {queue}: {e}")

    def publish_message(self, queue: str, message: dict):
        try:
            self.channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2),
            )
        except pika.exceptions.AMQPError as e:
            print(f"Failed to publish message to {queue}: {e}")