from channels.generic.websocket import AsyncWebsocketConsumer
import json
import asyncio

class YourConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        # Process the received message here

        await self.send(json.dumps({
            'progress': current_progress_percentage
        }))

"""
class ExplainabilityConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        if text_data_json.get('command') == 'start_explainability':
            await self.run_explainability_process()

    async def run_explainability_process(self):
        # Your explainability logic here
        # You might need to adapt your synchronous code to asynchronous code
        # For example, use asyncio.sleep() to simulate long-running processes

        # Send progress updates back to the client
        for i in range(10):
            await asyncio.sleep(3)  # Simulating a task
            progress = (i + 1) * 10
            await self.send(json.dumps({'progress': progress}))

            if progress == 100:
                await self.send(json.dumps({'status': 'completed'}))
"""