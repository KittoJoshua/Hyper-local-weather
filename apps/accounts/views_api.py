"""
REST API Views for User Authentication & Token Generation.
"""
from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from apps.accounts.serializers import UserRegistrationSerializer, UserLoginSerializer, UserProfileSerializer
from apps.accounts.models import User


class RegisterAPIView(generics.CreateAPIView):
    """
    POST: Register a new Smallholder Farmer or Administrator.
    Returns auth token on registration.
    """
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            "message": "User registered successfully",
            "token": token.key,
            "user": UserProfileSerializer(user).data
        }, status=status.HTTP_201_CREATED)


class LoginAPIView(generics.GenericAPIView):
    """
    POST: Authenticate user with username/phone and password.
    Returns active auth token and user profile context.
    """
    serializer_class = UserLoginSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            "message": "Login successful",
            "token": token.key,
            "user": UserProfileSerializer(user).data
        }, status=status.HTTP_200_OK)


class ProfileAPIView(generics.RetrieveUpdateAPIView):
    """
    GET/PUT: View or update authenticated user profile.
    """
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user
