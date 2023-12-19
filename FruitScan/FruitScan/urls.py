#Contributors: Erik, Mijin, Patricia, Jonathan

"""
URL configuration for FruitScan project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# Import relevant packages
from django.contrib import admin
from fruitscanapp.admin import admin_site
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

# Define url patterns
urlpatterns = [
    path('admin/', admin.site.urls),
    # This is a path to the admin login which then redirects us to our custom admin panel
    path('custom_admin/', admin_site.urls),
    path("", include("fruitscanapp.urls")),
    path('fruitscanapp/', include('fruitscanapp.urls')),
    path('accounts/', include('fruitscanapp.urls')),
    path('login/', auth_views.LoginView.as_view(), name='login'),
]

# Define url patterns in production mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
