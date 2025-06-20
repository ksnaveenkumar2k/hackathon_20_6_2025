
from django.urls import path
from . import views

app_name = 'api'  # Added for namespacing

urlpatterns = [
    path('profile/', views.StudentProfileView.as_view(), name='profile'),  # Changed name to 'profile' for consistency
]