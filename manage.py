#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finalProject.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


# ***********************************************************************************************************************
import os, sys, django

# add the folder that contains manage.py
sys.path.append(r"D:\Projects\bigDataFinalProject\finalProject")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finalProject.settings")
django.setup()

from primary.utils import load_model

m_rf = load_model("primary_rf")
m_lr = load_model("primary_logreg")

# ***********************************************************************************************************************


if __name__ == '__main__':
    main()
