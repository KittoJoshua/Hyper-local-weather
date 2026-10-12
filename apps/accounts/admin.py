"""
Admin configuration for custom User model.
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from apps.accounts.models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("username", "phone_number", "email", "role", "county", "is_staff")
    list_filter = ("role", "county", "is_staff", "is_active")
    fieldsets = BaseUserAdmin.fieldsets + (
        ("Micro-Climate Profile", {"fields": ("phone_number", "role", "county")}),
    )
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ("Micro-Climate Profile", {"fields": ("phone_number", "role", "county")}),
    )
