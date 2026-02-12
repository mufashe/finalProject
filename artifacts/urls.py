# artifacts/urls.py
from django.urls import path
from . import views
from .views import upload_and_predict, predict_one, predict_one_ui, dashboard_data_api, dashboard_page, \
    prediction_files_api, filters_api, upload_and_predict_alevel, predict_one_alevel, predict_one_alevel_ui, \
    alevel_dashboard_page, alevel_prediction_files_api, alevel_filters_api, alevel_dashboard_data_api

app_name = "artifacts"

urlpatterns = [
    path("", views._collect_uploads, name="list"),  # /artifacts/
    path("run/", views.run_inference, name="run"),  # /artifacts/run/
    path("predict/", upload_and_predict, name="upload_and_predict"),
    path("predict_one/", predict_one, name="predict_one"),
    path("predict_one_ui/", predict_one_ui, name="predict_one_ui"),
    path("dashboard/", dashboard_page, name="dashboard_page"),
    path("api/dashboard-data/", dashboard_data_api, name="dashboard_data_api"),
    path("api/prediction-files/", prediction_files_api, name="prediction_files_api"),
    path("api/filters/", filters_api, name="filters_api"),

    path("uploadal/", upload_and_predict_alevel, name="upload_and_predict_alevel"),
    path("predict_one_al/", predict_one_alevel, name="predict_one_alevel"),
    path("predict_one/ui_al/", predict_one_alevel_ui, name="predict_one_alevel_ui"),

    path("aldashboard/", alevel_dashboard_page, name="alevel_dashboard_page"),
    path("alapi/files/", alevel_prediction_files_api, name="alevel_prediction_files_api"),
    path("alapi/filters/", alevel_filters_api, name="alevel_filters_api"),
    path("alapi/data/", alevel_dashboard_data_api, name="alevel_dashboard_data_api"),

]
