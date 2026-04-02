#!/usr/bin/env bash
# Source this file in your shell:
#   source ./scripts/dev_aliases.sh

alias ds-all-up='./scripts/dev_stack.sh up --detach-app'
alias ds-all-down='./scripts/dev_stack.sh down'
alias ds-force-down='./scripts/dev_stack.sh force-down'
alias ds-force-down-now='./scripts/dev_stack.sh force-down --yes'
alias ds-all-cli-up='./scripts/dev_stack.sh up --detach-app --app-cmd "./.venv/bin/python -m app.cli.main --interactive --session-id dev-stack"'

alias ds-llm-up='./scripts/dev_stack.sh llm-up'
alias ds-llm-down='./scripts/dev_stack.sh llm-down'

alias ds-docker-up='./scripts/dev_stack.sh docker-up'
alias ds-docker-down='./scripts/dev_stack.sh docker-down'

alias ds-app-up='./scripts/dev_stack.sh app-up'
alias ds-app-down='./scripts/dev_stack.sh app-down'
alias ds-app-cli-up='./scripts/dev_stack.sh app-up --app-cmd "./.venv/bin/python -m app.cli.main --interactive --session-id dev-stack"'

alias ds-status='./scripts/dev_stack.sh status'
alias ds-logs='./scripts/dev_stack.sh logs all'
