"""
Custom User and Profile Models for Smallholder Farmers and Administrators.
"""
from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """
    Custom user model with telephone authentication and role division.
    Roles: Smallholder Farmer, System Administrator.
    """
    class Role(models.TextChoices):
        FARMER = "FARMER", "Smallholder Farmer"
        ADMIN = "ADMIN", "System Administrator"

    phone_number = models.CharField(max_length=20, unique=True, db_index=True)
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.FARMER)
    county = models.CharField(max_length=100, blank=True, null=True, help_text="Primary county of farming operations")
    created_at = models.DateTimeField(auto_now_add=True)

    REQUIRED_FIELDS = ["email", "phone_number"]

    def __str__(self):
        return f"{self.username} ({self.get_role_display()}) - {self.phone_number}"

    @property
    def is_farmer(self):
        return self.role == self.Role.FARMER

    @property
    def is_platform_admin(self):
        return self.role == self.Role.ADMIN or self.is_superuser
