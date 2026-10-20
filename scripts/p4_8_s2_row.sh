#!/bin/bash
# P4-8 deck item (6): when S2 @120K's standard report lands (speckle + unseen JSONs, written by the S2 120K report
# watcher), add its row and pre-registered verdict to outputs/p4_8/primary_table.md and push ONLY those files.
# S2 is NOT added to the meeting_v2 figures.  Usage: setsid nohup bash scripts/p4_8_s2_row.sh >> logs/watch_p4_8_s2_row.log 2>&1 &
set -u
cd /fs/nexus-projects/multimodal_recon/adaptable-acoustic-fields
until [ -f outputs/p4_5/speckle/S2_120K.json ] && [ -f outputs/p4_8/unseen/S2_120K.json ] && \
      [ -f outputs/p4_5/train_offlat_fit/S2_120K.json ]; do sleep 300; done
sleep 60
python3 scripts/p4_8_primary_table.py > /dev/null
V=$(python3 -c "import json; v=json.load(open('outputs/p4_8/primary_table.json'))['verdicts'].get('S2_120K', {}); print('HOLDS' if v.get('holds') else 'FAILS')")
F="outputs/p4_8/primary_table.md outputs/p4_8/primary_table.json outputs/p4_8/unseen/S2_120K.json outputs/p4_5/speckle/S2_120K.json outputs/p4_5/train_offlat_fit/S2_120K.json"
for f in outputs/p4_8/GAP_CI_S2_120K.json outputs/p4_8/RISK_METRICS_S2_120K.json outputs/p4_6/family_oof/S2_120K.json \
         outputs/p4_5/gate2_dense/S2_120K__track_p4_5_testdense.json; do [ -f "$f" ] && F="$F $f"; done
git add $F && git commit -q -m "P4-8 S2 @120K: row and pre-registered verdict ($V) added to the primary table (UF >= 0.510 AND HF >= 0.484)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $F && git push -q origin main
echo "$(date +%m-%d_%H:%M) S2_120K verdict $V; pushed $(git log --oneline -1)"
