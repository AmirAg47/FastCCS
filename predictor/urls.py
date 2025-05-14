from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect 
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('upload/', views.upload, name='upload'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
 
 
]

if settings.DEBUG:
    from django.conf import settings
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)