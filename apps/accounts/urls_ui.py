"""
UI route definitions for accounts and home redirects.
"""
from django.urls import path
from apps.accounts.views_ui import login_page, register_page, logout_page

urlpatterns = [
    path('login/', login_page, name='login'),
    path('register/', register_page, name='register'),
    path('logout/', logout_page, name='logout'),
]
