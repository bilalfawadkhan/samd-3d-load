#!/bin/bash
set -e

# Set workspace directory
WORKSPACE_DIR="${WORKSPACE_DIR:-/app}"

# Start nginx service (often used by RunPod as proxy)
start_nginx() {
    echo "Starting Nginx service..."
    service nginx start
}

# Setup ssh (Crucial for pod/dev mode)
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        # Generate SSH host keys if not present
        if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
            ssh-keygen -A
        fi
        service ssh start
        echo "SSH host keys:"
        cat /etc/ssh/*.pub
    else
        echo "No PUBLIC_KEY found. SSH not configured."
    fi
}

# Export env vars for SSH sessions
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Call Python handler (Serverless Mode)
call_python_handler() {
    echo "Calling Python handler.py..."
    cd "$WORKSPACE_DIR"
    python -u handler.py
}

# Dev environment (Pod Mode)
start_dev_mode() {
    echo "Starting in Dev Mode (Pod Mode)..."
    # We sleep infinity to keep the pod alive for SSH/Web Terminal access
    sleep infinity
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

start_nginx
setup_ssh
export_env_vars

# Determine mode: Check ENV or Fallback
if [[ "$DEV_MODE" == "true" ]] || [[ "$MODE_TO_RUN" == "pod" ]]; then
    start_dev_mode
else
    echo "Running in serverless mode (default)"
    call_python_handler
fi
