"""
Custom User and Authentication Models for the Hyper-Local Micro-Climate Platform.

Extends Django's AbstractUser to provide:
- Role-based authorization: Smallholder Farmer vs. System Administrator
- Primary telephone authentication tailored for smallholder farmers in Kenya
- Geographic county tracking for regional agro-climatic advisory filtering
"""
from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """
    Custom user model with telephone authentication and role division.
    
    Roles:
      - FARMER: Registered smallholder farmer managing one or more farm plots.
      - ADMIN: Platform administrator with access to ML models and station feeds.
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
