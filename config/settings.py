"""Настройки Django. Секреты и пути берутся из окружения, под одну машину ничего не захардкожено."""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MEDIA_ROOT = Path(os.environ.get("FSL_MEDIA_ROOT", BASE_DIR / "media"))


def _secret_key() -> str:
    """FSL_SECRET_KEY из окружения или ключ, сгенерированный один раз в media/ (не в git)."""
    if key := os.environ.get("FSL_SECRET_KEY"):
        return key
    p = MEDIA_ROOT / ".secret_key"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_urlsafe(50), encoding="ascii")
    return p.read_text(encoding="ascii").strip()


# Локальный учебный запуск. Для деплоя задать FSL_SECRET_KEY и FSL_DEBUG=0.
SECRET_KEY = _secret_key()
DEBUG = os.environ.get("FSL_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("FSL_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "analyzer",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"},
}

# Сессия хранит только случайный ID папки с записями: БД для этого не нужна.
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"
SESSION_COOKIE_HTTPONLY = True

LANGUAGE_CODE = "ru"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Рабочие файлы: записи по сессиям
SESSIONS_ROOT = MEDIA_ROOT / "sessions"
SAMPLES_DIR = BASE_DIR / "samples"

DATA_UPLOAD_MAX_MEMORY_SIZE = 2_621_440
FILE_UPLOAD_MAX_MEMORY_SIZE = 2_621_440
FSL_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
