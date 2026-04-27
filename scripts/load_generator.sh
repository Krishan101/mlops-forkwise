#!/usr/bin/env bash
# =============================================================================
# ForkWise Load Generator
# =============================================================================
# Generates continuous emulated user traffic against the substitution API.
# Simulates realistic usage: substitution queries + accept/reject feedback.
#
# Usage:
#   bash scripts/load_generator.sh                          # defaults: 24h, 0.5 req/s
#   bash scripts/load_generator.sh --duration 3600          # 1 hour
#   bash scripts/load_generator.sh --rate 2.0               # 2 req/s
#   bash scripts/load_generator.sh --api-host <IP>:30808    # custom host
#
# Run in tmux for long-running operation:
#   tmux new -s loadgen
#   bash scripts/load_generator.sh --duration 604800
#   # Ctrl+B then D to detach
# =============================================================================

set -uo pipefail

GREEN=$'\e[32m' YELLOW=$'\e[33m' RESET=$'\e[0m'
log()  { echo -e "${GREEN}[loadgen]${RESET} $(date +%H:%M:%S) $*"; }
warn() { echo -e "${YELLOW}[loadgen]${RESET} $(date +%H:%M:%S) $*"; }

# Parse arguments
DURATION=86400
RATE=0.5
API_HOST="192.168.1.11:30808"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --rate)     RATE="$2"; shift 2 ;;
        --api-host) API_HOST="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

API="http://$API_HOST"

# Realistic ingredient-recipe pairs
INGREDIENTS=("butter" "eggs" "milk" "flour" "sugar" "salt" "vanilla extract"
             "chocolate chips" "brown sugar" "baking powder" "baking soda"
             "soy sauce" "sesame oil" "chicken breast" "garlic" "ginger"
             "vegetable oil" "broccoli" "bell pepper" "cornstarch")
RECIPES=("Classic Pancakes" "Chocolate Chip Cookies" "Chicken Stir Fry"
         "Banana Bread" "Caesar Salad" "Tomato Soup" "Pasta Carbonara"
         "Grilled Cheese" "Fried Rice" "Omelette")

TOTAL=0
ACCEPTED=0
REJECTED=0
ERRORS=0
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))

log "Starting load generator"
log "  API:      $API"
log "  Duration: ${DURATION}s"
log "  Rate:     ${RATE} req/s"
log "  End time: $(date -d @$END_TIME 2>/dev/null || date -r $END_TIME 2>/dev/null || echo 'N/A')"
log ""

while [[ $(date +%s) -lt $END_TIME ]]; do
    # Pick random ingredient and recipe
    ING="${INGREDIENTS[$((RANDOM % ${#INGREDIENTS[@]}))]}"
    RECIPE="${RECIPES[$((RANDOM % ${#RECIPES[@]}))]}"

    # Send substitution query
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API/substitute" \
        -H "Content-Type: application/json" \
        -d "{\"ingredient\":\"$ING\",\"recipe_name\":\"$RECIPE\",\"top_k\":3}" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" == "200" ]]; then
        TOTAL=$((TOTAL + 1))
        QUERY_ID=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('query_id',''))" 2>/dev/null)
        SUGGESTIONS=$(echo "$BODY" | python3 -c "import sys,json; s=json.load(sys.stdin).get('suggestions',[]); print(len(s))" 2>/dev/null)

        # Randomly send feedback (70% of the time)
        if [[ -n "$QUERY_ID" && "$SUGGESTIONS" -gt 0 && $((RANDOM % 10)) -lt 7 ]]; then
            # Pick first suggestion
            SUGGESTED=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['suggestions'][0]['ingredient'])" 2>/dev/null)

            # 60% accept, 40% reject
            if [[ $((RANDOM % 10)) -lt 6 ]]; then
                ACCEPT="true"
                ACCEPTED=$((ACCEPTED + 1))
            else
                ACCEPT="false"
                REJECTED=$((REJECTED + 1))
            fi

            curl -s -X POST "$API/feedback" \
                -H "Content-Type: application/json" \
                -d "{\"request_id\":\"$QUERY_ID\",\"recipe_id\":\"loadgen\",\"missing_ingredient\":\"$ING\",\"suggested_substitution\":\"$SUGGESTED\",\"user_accepted\":$ACCEPT}" \
                > /dev/null 2>&1
        fi

        # Log every 10 requests
        if [[ $((TOTAL % 10)) -eq 0 ]]; then
            ELAPSED=$(( $(date +%s) - START_TIME ))
            REMAINING=$(( END_TIME - $(date +%s) ))
            log "  requests=$TOTAL accepted=$ACCEPTED rejected=$REJECTED errors=$ERRORS elapsed=${ELAPSED}s remaining=${REMAINING}s"
        fi
    else
        ERRORS=$((ERRORS + 1))
        if [[ $((ERRORS % 5)) -eq 0 ]]; then
            warn "  $ERRORS errors so far (last: HTTP $HTTP_CODE)"
        fi
    fi

    # Sleep with some jitter for realistic traffic pattern
    SLEEP=$(python3 -c "import random; print(random.expovariate($RATE))" 2>/dev/null || echo "2")
    # Cap sleep at 30s to avoid long gaps
    SLEEP=$(python3 -c "print(min($SLEEP, 30.0))")
    sleep "$SLEEP"
done

# Final summary
ELAPSED=$(( $(date +%s) - START_TIME ))
log ""
log "============================================"
log "  Load generation complete"
log "============================================"
log "  Duration:   ${ELAPSED}s"
log "  Requests:   $TOTAL"
log "  Accepted:   $ACCEPTED"
log "  Rejected:   $REJECTED"
log "  Errors:     $ERRORS"
log "  Avg rate:   $(python3 -c "print(f'{$TOTAL / max($ELAPSED,1):.2f}') " 2>/dev/null) req/s"
log "============================================"
