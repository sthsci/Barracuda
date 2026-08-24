# Optional BARRACUDA account service

The existing Dash application remains the only public user interface. It runs
on <http://127.0.0.1:8501> and keeps the established scientific page structure.

The Django service in `api/` is optional. It adds account registration, saved
CSV datasets and expiring read-only spreadsheet links. Users can ignore it and
continue using every Dash analysis without an account. Raw microscopy is out
of scope and is never accepted by this service.

## Local development

For the normal account-free application, use the established command from the
repository root:

```bash
.venv/bin/python dash_app.py
```

To test optional accounts and CSV sharing, start Django in a second terminal:

```bash
cd platform/api
../../.venv/bin/python manage.py migrate
../../.venv/bin/python manage.py runserver 127.0.0.1:8000
```

Then open <http://127.0.0.1:8501/workspace> or choose **Account and CSV
sharing** in the Dash sidebar. Local development uses SQLite and private filesystem storage under
`platform/api/var/`.

For an integrated local infrastructure test, copy `.env.example` to `.env`
and replace every `change-me` value, then run:

```bash
docker compose up --build
```

The Compose stack starts the same Dash interface on port 8501, plus PostgreSQL,
Redis, private MinIO storage, Django, the scientific worker and hourly expiry
cleanup. It does not replace the established analysis pages.

## Privacy boundary

- Guest Dash use remains account-free and nothing is saved by this service.
- Saving happens only after a user signs in and presses **Save current CSV**.
- Spreadsheet links are read-only, expiring and revocable.
- Raw CSV download is disabled unless an account owner explicitly enables it.
- Imaging files, identifiers and clinical metadata are not supported.
