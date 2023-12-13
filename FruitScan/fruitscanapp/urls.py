from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .views import register, user_logout, CustomLoginView

urlpatterns = [
    path("", views.home, name="home"),
    path("admin_view", views.adminPanel, name="admin_view"),
    path('train_model/', views.train_model_view, name='train_model'),
    path('register/', register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('user_logout/', user_logout, name='user_logout'),
    path('login/', CustomLoginView.as_view(), name='login'),    
    path('upload_zip', views.upload_zip, name="upload_zip"),
    path('upload_test_set', views.upload_test_set, name='upload_test_set'),
    path('delete_all_images/', views.delete_all_images, name='delete_all_images'),
    path('delete_test_set/', views.delete_test_set, name='delete_test_set'),
    path('train_model_view/', views.train_model_view, name='train_model_view'),
    path('deploy_selected', views.deploy_selected, name='deploy_selected'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('test_deployed_model',views.test_deployed_model, name='test_deployed_model'),
    path('explain', views.explainability, name='explainability'),
    ]
