import os
import json
import asyncio
from typing import Dict
from aio_pika import connect_robust, Message, DeliveryMode, RobustChannel, RobustConnection
from aio_pika.exceptions import AMQPException


class RabbitMQManager:
    def __init__(self, host: str, username: str, password: str):
        self.host = host
        self.username = username
        self.password = password
        self.message_ttl = int(os.getenv("MESSAGE_TTL", 10000))
        self.max_length = int(os.getenv("MAX_LENGTH", 1000))
        self.connection: RobustConnection | None = None
        self.channel: RobustChannel | None = None
        self.lock = asyncio.Lock()  # Ensure thread-safe connection setup

    async def connect(self):
        async with self.lock:
            if self.connection and not self.connection.is_closed:
                return
            self.connection = await connect_robust(
                host=self.host,
                login=self.username,
                password=self.password
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)

    async def create_queue(self, queue: str):
        await self.connect()
        try:
            await self.channel.declare_queue(
                queue,
                durable=True,
                arguments={
                    "x-message-ttl": self.message_ttl,
                    "x-max-length": self.max_length,
                    "x-overflow": "drop-head",
                    "x-consumer-timeout": 10000,
                }
            )
        except AMQPException as e:
            raise RuntimeError(f"Failed to declare queue '{queue}': {e}")

    async def publish_message(self, queue: str, message: Dict):
        await self.connect()
        try:
            await self.create_queue(queue)  # ensure queue exists before sending
            await self.channel.default_exchange.publish(
                Message(
                    body=json.dumps(message).encode("utf-8"),
                    delivery_mode=DeliveryMode.PERSISTENT
                ),
                routing_key=queue
            )
        except AMQPException as e:
            print(f"[RabbitMQ] Failed to publish to {queue}: {e}")
