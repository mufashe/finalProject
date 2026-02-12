from django.urls import path

from eduforecast.views import predictALevel, upload_excel, list_datasets, upload_dataset, dataset_detail, \
    process_dataset, processed_detail, all_datasets

# app_name = "uploader"
app_name = "pred"
urlpatterns = [
    path('alpre/', predictALevel, name='alpre'),
    # path('upload/', upload_excel, name='upload_excel'),
    path("", list_datasets, name="list"),
    path("upload/", upload_dataset, name="upload"),
    path("<int:pk>/", dataset_detail, name="detail"),
    path("<int:pk>/process/", process_dataset, name="process"),
    path("<int:pk>/processed/", processed_detail, name="processed_detail"),
    path('datasets/', all_datasets, name='datasets'),

]
