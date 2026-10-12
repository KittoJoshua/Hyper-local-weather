"""
UI route definitions for farms.
"""
from django.urls import path
from apps.farms.views_ui import register_plot_page, plot_list_page

urlpatterns = [
    path('register/', register_plot_page, name='register_plot'),
    path('my-plots/', plot_list_page, name='my_plots'),
]
