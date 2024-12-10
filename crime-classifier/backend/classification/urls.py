from django.urls import path
from .views import crime_classification_view

urlpatterns = [
    path('predict/', crime_classification_view, name='predict'),
]