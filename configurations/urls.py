# levels/urls.py
from django.urls import path
from . import views

app_name = "levels"

urlpatterns = [
    path("", views.levels_manage, name="list"),  # /levels/
    path("<int:pk>/edit/", views.level_edit, name="edit"),
    path("<int:pk>/delete/", views.level_delete, name="delete"),
]
