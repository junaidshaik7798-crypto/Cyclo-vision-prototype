"""Outbound HTTPS helpers.

Windows machines that once had PostgreSQL/EDB installed often keep a
machine-wide ``CURL_CA_BUNDLE`` (or ``REQUESTS_CA_BUNDLE``/``SSL_CERT_FILE``)
that points at a CA file which no longer exists. ``requests`` honours those
variables, so *every* HTTPS call fails with::

    OSError: Could not find a suitable TLS CA certificate bundle, invalid path

That single stale variable is enough to make each IBTrACS dataset refresh
fail. Passing an explicit, verified bundle path overrides the environment, so
downloads work again without the user having to touch their system settings.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("cyclo.net")

# Order mirrors requests/urllib3 plus OpenSSL's own variable.
CA_ENV_VARS = (
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
    "DEFAULT_CA_BUNDLE_PATH",
)


@lru_cache(maxsize=1)
def ca_bundle() -> bool | str:
    """Return a usable ``requests`` ``verify`` target.

    Prefers an explicitly configured bundle *when the file actually exists*,
    then certifi, then the platform default (``True``). It never returns a
    path that is missing -- returning ``True`` instead means requests falls
    back to the OS trust store rather than raising ``OSError``.
    """
    for name in CA_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        if Path(value).is_file():
            return value
        logger.warning(
            "%s points at a missing CA bundle (%s); ignoring it and using the "
            "bundled certificate store instead.",
            name,
            value,
        )

    try:
        import certifi

        bundle = certifi.where()
        if Path(bundle).is_file():
            return bundle
    except Exception:  # noqa: BLE001 - certifi is optional at runtime
        logger.debug("certifi unavailable; using the platform default CA store.")

    return True
