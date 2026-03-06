#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="$SCRIPT_DIR/.secrets"
SENTINEL_FILE="$SCRIPT_DIR/.session_invalid"
CSV_FILE="$SCRIPT_DIR/NR8E_QRZ_stats.csv"
LOG_FILE="$SCRIPT_DIR/qrzTracker.log"

log_msg() {
    local level="$1"
    local msg="$2"
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "[$ts] [$level] $msg" | tee -a "$LOG_FILE"
}

notify() {
    local msg="$1"
    local bus="/run/user/$(id -u)/bus"
    if command -v notify-send &>/dev/null && [ -S "$bus" ]; then
        DBUS_SESSION_BUS_ADDRESS="unix:path=$bus" notify-send "QRZ Tracker" "$msg" 2>/dev/null || true
    fi
}

# Check for session invalid sentinel
if [ -f "$SENTINEL_FILE" ]; then
    log_msg "WARN" "Session marked invalid. Update .secrets and delete .session_invalid to resume."
    notify "QRZ session expired — update .secrets and delete .session_invalid"
    exit 0
fi

# Load secrets
if [ ! -f "$SECRETS_FILE" ]; then
    log_msg "ERROR" ".secrets file not found at $SECRETS_FILE"
    exit 1
fi
# shellcheck source=.secrets
source "$SECRETS_FILE"

if [ -z "${QRZ_SESSION_TOKEN:-}" ]; then
    log_msg "ERROR" "QRZ_SESSION_TOKEN is not set in .secrets"
    exit 1
fi

if [ -z "${QRZ_CALLSIGN:-}" ]; then
    log_msg "ERROR" "QRZ_CALLSIGN is not set in .secrets"
    exit 1
fi

# Fetch QRZ page
RESPONSE=$(curl --silent \
    "https://www.qrz.com/db/${QRZ_CALLSIGN}" \
    -H "Cookie: xf_session=${QRZ_SESSION_TOKEN}" \
    -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0")

# Extract lookup count
COUNT=$(echo "$RESPONSE" | grep -oP '(?<=Lookups: )[\d,]+' | tr -d ',')

if [ -z "$COUNT" ]; then
    log_msg "ERROR" "Failed to extract Lookups count — session may be expired"
    touch "$SENTINEL_FILE"
    notify "QRZ session expired — update .secrets and delete .session_invalid"
    exit 1
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "${TIMESTAMP},${COUNT}" >> "$CSV_FILE"
log_msg "INFO" "Recorded: ${COUNT} lookups at ${TIMESTAMP}"

python3 "$SCRIPT_DIR/qrzHitsViz.py"
