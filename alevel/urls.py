from django.urls import path

from alevel import views

app_name = 'alevel'
urlpatterns = [
    path('upload/', views.upload_a_level_dataset, name='upload'),
    path('uploads/', views.viewUploadedDataset, name='viewdataset'),
]
