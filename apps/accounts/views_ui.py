"""
Web UI Views for Farmer Authentication and Navigation.
"""
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from apps.accounts.models import User


def login_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        identifier = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=identifier, password=password)
        if not user:
            try:
                user_obj = User.objects.get(phone_number=identifier)
                user = authenticate(request, username=user_obj.username, password=password)
            except User.DoesNotExist:
                pass

        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.first_name or user.username}!")
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid telephone/username or password.")

    return render(request, 'accounts/login.html')


def register_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username')
        phone_number = request.POST.get('phone_number')
        email = request.POST.get('email')
        password = request.POST.get('password')
        county = request.POST.get('county')
        first_name = request.POST.get('first_name', '')
        last_name = request.POST.get('last_name', '')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username is already taken.")
        elif User.objects.filter(phone_number=phone_number).exists():
            messages.error(request, "A user with this phone number already exists.")
        else:
            user = User.objects.create_user(
                username=username,
                phone_number=phone_number,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                county=county,
                role=User.Role.FARMER
            )
            login(request, user)
            messages.success(request, "Registration successful! Welcome to the Micro-Climate Platform.")
            return redirect('dashboard')

    return render(request, 'accounts/register.html')


def logout_page(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')
