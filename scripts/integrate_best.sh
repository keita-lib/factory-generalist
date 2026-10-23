#!/bin/bash
# 統合（製品テーブル → ステップ → 排斥テーブルで取り上げ → 戻って置く）を、最良の設定で走らせる（2026-10-18）。
# 3 往復の完了 97.3 %（1024 台 × 乱数シード 1〜3：97.7 / 96.8 / 97.5 %）、3 往復 167 秒
# （10/17 の設定 scripts/integrate_1017.sh は 96.2 %・179 秒、10/16 の設定は 95.1 %・240 秒）。
# 1 往復の完了 99.0 %（乱数シード 1〜3：98.8 / 99.1 / 98.9 %）、1 往復 55.9 秒。
#
# 方策は段階ごとに切り替える（すべて checkpoints/ に同梱）：
#   歩く・寄る・下がる・降りる       run14（loco_run14_model_24493.pt）
#   ステップを昇る（排斥テーブルへ）   run30（climb_run30_model_24692.pt：run14 を段 20 cm 固定で 200 回追加学習）
#   取り上げ                           run28（pick_run28_model_30500.pt）
#   置いた後の振り返り（その場で回る） 回れ右の方策（turn_v1_model_24700.pt：run14 を「その場で回る」だけで 300 回追加学習）
#   置く                               run14（脚）＋ 逆運動学の決まった手順（腕）
#                                      ＋ 置くの残差の方策（place_residual_res2.pt：手順の腕の目標への小さな補正と、開くタイミング。
#                                        src/fg/place_residual.py、integrate.py の --residual-train で学習）
# 10/17 の設定からの違い：置くの残差の方策（--residual-policy）を足した
# 10/16 の設定からの違い：昇る間だけ run30、置いた後の振り返りに回れ右の方策（最大 1.0 rad/s）、
# 降りる 0.3 → 0.4 m/s、置くで下ろす時間 2.0 → 1.2 秒
#
# 使い方（リポジトリの一番上で。Windows は Git Bash）:
#   bash scripts/integrate_best.sh [台数] [往復] [乱数シード] [追加の引数...]
# 例:
#   bash scripts/integrate_best.sh 1024 3 1                               # 3 往復の完了率を測る
#   bash scripts/integrate_best.sh 1024 1 1 --metrics out/metrics.json    # 段階ごとの時間と滑らかさも測る
#   bash scripts/integrate_best.sh 32 3 21 --replay-video out/3trips.mp4  # 完了した 1 台を動画にする
cd "$(dirname "$0")/.."
N=${1:-1024}; T=${2:-3}; SEED=${3:-1}; shift 3 2>/dev/null || shift $#

# venv の python（Windows と Linux で場所が違う。PYTHON で差し替えられる）
if [ -n "$PYTHON" ]; then PY=$PYTHON
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=.venv/bin/python; fi

# 寄るときに 5 cm 手前で止める・製品テーブルの天板の奥行き・受け具の押さえのすき間
export PYTHONIOENCODING=utf-8 FG_APPROACH_BACK=0.05 FG_PROD_TOP_Y=0.30 FG_KEEP_GAP=0.006

exec "$PY" src/fg/integrate.py \
  --loco checkpoints/loco_run14_model_24493.pt --manip checkpoints/pick_run28_model_30500.pt \
  --climb-policy checkpoints/climb_run30_model_24692.pt --turn-policy checkpoints/turn_v1_model_24700.pt --turn-wmax 1.0 \
  --trips $T --num-envs $N --seconds $((T * 110 + 30)) --seed $SEED \
  --place-scripted --place-closed --prod-top 0.952 --place-dz -0.01 --place-low-dur 1.2 \
  --fresh-obs --reset-hist --filt-actual --settle 0.05 --tol-prod 0.02 --dead-frac 0.5 \
  --settle-correct --place-max-sel 1 --back-prod 0.5 --vx-down 0.4 \
  --pick-dxy 0.08 --place-hold 0.02 --place-hold-ph1 --place-hold-box --pick-hook-tol 0.03 --settle-qvel 1.0 --carry-delay 1.5 \
  --residual-policy checkpoints/place_residual_res2.pt   --open-hold --align-first 0.05 --align-back 0.25 --fine-kick 10 --step-novy --pick-retry 8 --retry-wait 1.5 \
  "$@"
