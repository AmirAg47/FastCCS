from __future__ import absolute_import, unicode_literals

# برای اطمینان از بارگذاری Celery در هنگام شروع پروژه
from .celery import app as celery_app

__all__ = ('celery_app',)
