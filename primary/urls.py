from django.urls import path

from primary import views

app_name = 'primary'
urlpatterns = [
    path('upload/', views.upload_primary_excel, name='upload'),
    path('uploads/', views.uploads_list, name='uploads_list'),

    path("up/", views.upload_and_predict, name="upload_predict"),
    path("predictions/", views.predictions_list, name="predictions_list"),
    path("predictions/<int:batch_id>/", views.prediction_detail, name="prediction_detail"),
]
