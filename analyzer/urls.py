from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/upload", views.api_upload),
    path("api/sample", views.api_sample),
    path("api/rec/<str:rid>", views.api_overview),
    path("api/rec/<str:rid>/audio.wav", views.rec_audio),
    path("api/rec/<str:rid>/step1", views.api_step1),
    path("api/rec/<str:rid>/step1.wav", views.step1_audio),
    path("api/rec/<str:rid>/series", views.api_series),
    path("api/rec/<str:rid>/coef", views.api_coef),
    path("api/rec/<str:rid>/spectrum", views.api_spectrum),
    path("api/rec/<str:rid>/edit", views.api_edit),
    path("api/rec/<str:rid>/edit.wav", views.edit_audio),
    path("api/rec/<str:rid>/band.wav", views.band_audio),
    path("api/rec/<str:rid>/spectrogram", views.api_spectrogram),
]
