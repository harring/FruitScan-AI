from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from fruitscanapp import consumers  # Import your WebSocket consumer
from fruitscanapp.consumers import ExplainabilityConsumer

application = ProtocolTypeRouter({
    "websocket": URLRouter([
        path("ws/somepath/", consumers.YourConsumer.as_asgi()),  # WebSocket URL path
    ]),
})

"""
websocket_urlpatterns = [
    path('ws/explainability/', ExplainabilityConsumer.as_asgi()),
]
"""