#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

python manage.py migrate --noinput
exec gunicorn barracuda_api.wsgi:application \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-2}" \
  --threads "${GUNICORN_THREADS:-4}" \
  --timeout "${GUNICORN_TIMEOUT:-120}" \
  --access-logfile - \
  --error-logfile -
