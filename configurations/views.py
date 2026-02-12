# levels/views.py
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.db.models import ProtectedError

from .models import EducationLevel
from .forms import EducationLevelForm


@require_http_methods(["GET", "POST"])
def levels_manage(request):
    """
    Combined list + create view.
    GET  -> show list and empty form
    POST -> create new level
    """
    if request.method == "POST":
        form = EducationLevelForm(request.POST)
        if form.is_valid():
            level = form.save()
            messages.success(request, f"Level “{level.name}” saved.")
            return redirect("levels:list")
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = EducationLevelForm()

    levels = EducationLevel.objects.order_by("id")
    return render(request, "levels.html", {
        "form": form,
        "levels": levels,
        "mode": "create",
    })


@require_http_methods(["GET", "POST"])
def level_edit(request, pk):
    level = get_object_or_404(EducationLevel, pk=pk)
    if request.method == "POST":
        form = EducationLevelForm(request.POST, instance=level)
        if form.is_valid():
            form.save()
            messages.success(request, "Level updated.")
            return redirect("levels:list")
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = EducationLevelForm(instance=level)

    levels = EducationLevel.objects.order_by("id")
    return render(request, "levels/levels_manage.html", {
        "form": form,
        "levels": levels,
        "mode": "edit",
        "edit_obj": level,
    })


@require_http_methods(["POST"])
def level_delete(request, pk):
    level = get_object_or_404(EducationLevel, pk=pk)
    name = level.name
    try:
        level.delete()
        messages.success(request, f"Level “{name}” deleted.")
    except ProtectedError:
        messages.error(
            request,
            f"Cannot delete “{name}” because it is referenced by other records."
        )
    except Exception as e:
        messages.error(request, f"Could not delete “{name}”: {e}")
    return redirect("levels:list")
