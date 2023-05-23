from django.urls import path
from . import views

app_name = 'process'

urlpatterns = [
    path('process/', views.HelloWorldView.as_view()),
    path('name/', views.NameView.as_view(), name='name'),
    path('image_size/', views.ImageSizeView.as_view(), name='image_size'),
    path('gesture/', views.ImagePredView.as_view(), name='gesture')
]