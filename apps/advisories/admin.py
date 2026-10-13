"""
Admin configuration for AdvisoryAlert model.
"""
from django.contrib import admin
from apps.advisories.models import AdvisoryAlert


@admin.register(AdvisoryAlert)
class AdvisoryAlertAdmin(admin.ModelAdmin):
    list_display = ("headline", "farmplot", "alert_type", "severity", "is_critical", "created_at")
    list_filter = ("alert_type", "severity", "is_critical")
    search_fields = ("headline", "agronomic_advice", "farmplot__name")
