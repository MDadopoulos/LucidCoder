#!/usr/bin/env bash
# entrypoint.sh — materialise the Vertex service-account JSON from the
# AgentBeats config secret, then exec the A2A server.
#
# AgentBeats provides the SA key as a single env var
# (GOOGLE_APPLICATION_CREDENTIALS_JSON). We write it to a file in the
# container's writable tmp dir and point google-genai's ADC chain at it.
set -euo pipefail

CRED_FILE="${GOOGLE_APPLICATION_CREDENTIALS:-/tmp/sa.json}"

if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
    umask 077
    printf '%s' "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > "$CRED_FILE"
    export GOOGLE_APPLICATION_CREDENTIALS="$CRED_FILE"
    unset GOOGLE_APPLICATION_CREDENTIALS_JSON
fi

if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
    echo "FATAL: GOOGLE_CLOUD_PROJECT is required for Vertex AI mode." >&2
    exit 2
fi

export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

exec python -m src.server
