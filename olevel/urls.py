from django.urls import path

from olevel import views
from olevel.views import predict_one_olevel_ui, upload_and_predict_olevel, predict_one_olevel, olevel_dashboard_page, \
    olevel_prediction_files_api, olevel_filters_api, olevel_dashboard_data_api

app_name = 'olevel'
urlpatterns = [
    path('upload/', views.upload_o_level_dataset, name='upload'),
    path('uploads/', views.viewUploadedDataset, name='viewdataset'),
    path("uploader/", views.upload_and_predict_olevel, name="upload_and_predict_olevel"),
    path("predict_one/", views.predict_one_olevel, name="predict_one_olevel"),
    path("predict_one/ui/", predict_one_olevel_ui, name="predict_one_olevel_ui"),
    path("dashboard/", olevel_dashboard_page, name="olevel_dashboard_page"),
    path("api/prediction_files/", olevel_prediction_files_api, name="olevel_prediction_files_api"),
    path("api/filters/", olevel_filters_api, name="olevel_filters_api"),
    path("api/data/", olevel_dashboard_data_api, name="olevel_dashboard_data_api"),

]
