from django.urls import path
from .views import index, predict_api


urlpatterns = [
    path('', index, name="index"),
    path("api/predict/", predict_api, name="predict_api"),
]