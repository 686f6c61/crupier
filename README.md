# Crupier Landing

Static production site for [crupier.686f6c61.dev](https://crupier.686f6c61.dev/).
This branch presents Crupier `0.6.0`; package source, installation guidance,
release notes, and developer documentation live on
[`main`](https://github.com/686f6c61/crupier/tree/main).

## Deployment Contract

- Source branch: `landing`
- Container entrypoint: `Dockerfile.site`
- Public assets: `site/`
- Runtime: Nginx on port `80`
- Deployment: automatic through the configured private GitHub integration

The container copies only the static site. Package source, tests, local
configuration, credentials, and the social-card source file are not exposed by
Nginx.

## Validate Locally

```bash
python3 site/validate.py
python3 "$HOME/.codex/skills/crear-landing-html/scripts/audit_landing.py" \
  site/index.html --strict
docker build -f Dockerfile.site -t crupier-landing:0.6.0 .
```

Open `site/index.html` directly for content and layout checks. Validate the
container before publishing changes that affect Nginx, headers, or deployment.

## Product Documentation

- [README](https://github.com/686f6c61/crupier/blob/main/README.md)
- [Changelog](https://github.com/686f6c61/crupier/blob/main/CHANGELOG.md)
- [PyPI](https://pypi.org/project/crupier/)

Do not store provider keys, deployment credentials, analytics secrets, or
private environment files in this branch.
