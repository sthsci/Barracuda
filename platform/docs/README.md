# BARRACUDA platform migration notes

These documents define the safe boundary between the staged web platform and
the existing Barracuda scientific runtime. They are intentionally implementation
oriented: decisions are stated as defaults that can be encoded in Django,
the worker image and Docker Compose.

- [Architecture and local deployment](architecture.md)
- [Guest privacy, retention and sharing](security-and-privacy.md)
- [Scientific worker and adapter contract](worker-adapter-contract.md)

The migration should preserve one rule above all others: the API and worker
orchestrate a versioned BARRACUDA package; they do not copy or reimplement the
PyMC models.

