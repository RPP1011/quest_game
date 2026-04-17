#!/bin/bash
# Overnight 2×2×10 rollout against the Pale Lights seed.
# Generates 3 candidates, picks the top 2, generates skeletons for both,
# then runs 4 rollouts (2 candidates × 2 profiles × 10 chapters each).
#
# Expected GPU time: ~4 hrs on Gemma 4 26B A4B at ~5 min/chapter.
# Resume-safe: if killed, re-run this script and each rollout picks up
# from its last committed chapter.
#
# Usage: bash tools/overnight_rollout.sh 2>&1 | tee /tmp/overnight_rollout.log

set -euo pipefail
QUESTS_DIR=data/quests
QID=pale_lights
SERVER=http://127.0.0.1:8082
CHAPTERS=10
PROFILES="impulsive cautious"

echo "=== Step 1: restart server with latest code ==="
lsof -i :8000 -t 2>/dev/null | xargs -r kill 2>/dev/null || true
sleep 2
uv run quest serve --quests-dir "$QUESTS_DIR" --server "$SERVER" --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
sleep 3
echo "server pid=$SERVER_PID"

echo "=== Step 2: generate candidates ==="
CIDS=$(curl -sf -X POST "http://127.0.0.1:8000/api/quests/$QID/candidates/generate?n=3" \
    | python3 -c "import sys,json; cs=json.load(sys.stdin); print(' '.join(c['id'] for c in cs))")
echo "candidates: $CIDS"
CID_ARRAY=($CIDS)
C1=${CID_ARRAY[0]}
C2=${CID_ARRAY[1]}
echo "picking top 2: $C1, $C2"

echo "=== Step 3: pick + skeleton for candidate 1 ($C1) ==="
curl -sf -X POST "http://127.0.0.1:8000/api/quests/$QID/candidates/$C1/pick" -o /dev/null
curl -sf -X POST "http://127.0.0.1:8000/api/quests/$QID/candidates/$C1/skeleton/generate" -o /dev/null
echo "  skeleton generated for $C1"

echo "=== Step 4: pick + skeleton for candidate 2 ($C2) ==="
curl -sf -X POST "http://127.0.0.1:8000/api/quests/$QID/candidates/$C2/pick" -o /dev/null
curl -sf -X POST "http://127.0.0.1:8000/api/quests/$QID/candidates/$C2/skeleton/generate" -o /dev/null
echo "  skeleton generated for $C2"

# Re-pick C1 so the quest's config.json points at C1 for rollout 1-2
curl -sf -X POST "http://127.0.0.1:8000/api/quests/$QID/candidates/$C1/pick" -o /dev/null

echo "=== Step 5: kill server (rollouts use CLI, not server) ==="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "=== Step 6: launch rollouts sequentially ==="
for CID in $C1 $C2; do
    for PROFILE in $PROFILES; do
        echo ""
        echo "--- rollout: candidate=$CID profile=$PROFILE chapters=$CHAPTERS ---"
        uv run quest rollout \
            --quests-dir "$QUESTS_DIR" \
            --quest "$QID" \
            --candidate "$CID" \
            --profile "$PROFILE" \
            --chapters "$CHAPTERS" \
            --server "$SERVER" \
            2>&1 || echo "  [WARN] rollout failed — will try next"
    done
done

echo ""
echo "=== Done ==="
echo "Rollouts:"
uv run python -c "
from app.world.db import open_db
from app.world.state_manager import WorldStateManager
sm = WorldStateManager(open_db('$QUESTS_DIR/$QID/quest.db'))
for r in sm.list_rollouts():
    print(f'  {r.id} candidate={r.candidate_id} profile={r.profile_id} '
          f'status={r.status.value} {r.chapters_complete}/{r.total_chapters_target}')
" 2>&1 | grep -v UserWarning | grep -v "class "
