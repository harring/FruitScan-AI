"""
WSGI config for FruitScan project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

# Import relevant packages
import os
from django.core.wsgi import get_wsgi_application

# Set default environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FruitScan.settings')

# Creating an ASGI (Asynchronous Server Gateway Interface) application object 
# for the Django project.
application = get_wsgi_application()
