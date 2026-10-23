"""統合のタスク：検査室の製品テーブル → 5 m 歩く → ステップ（20 cm × 2 段）を上る → 台の上の排斥テーブルでケースを取り上げる →
振り返ってステップを降りる → 製品テーブルに置く。これを 3 往復（作者の決定、2026-10-13）。

方策の切り替え（スキルの組み合わせ）
- 歩く・段差・寄る・離れる・振り返る：run14（速度の指令に従う歩行。段差 100 %、落ちる箱の運搬 100 %）
- 取り上げ・置く：V16 の方策（腕も方策が動かす。課題の入力 22 個）。V16 は平らな床だけで学習したので段差の力は失っている
  （20 cm 上り 0 %）→ 段差は run14 に任せる
- 2 つの方策の入力は、V16 が run14 の 280 個＋22 個。同じ環境（課題 V16）で両方を動かし、run14 には先頭の 280 個を渡す
進行役（どの方策を使うか、歩く速さの指令）はこのスクリプトが決める。箱・テーブルの位置は Phase 1 の扱い（シミュレーターの正解）

配置（区画の座標、原点＝台の中心・台の上面、床は z = −0.4 m）
- ステップは −x 側から上る（1 段目の縁 x = −1.35 m、台 −1.0〜+1.0 m）
- 排斥テーブル：台の上 x = +0.55 m（天板 946 mm、台の上面から）
- 製品テーブル：床 x = −6.35 m（1 段目の縁から 5 m、天板 938 mm、床から）
- 往復のたびに新しいケース：置いて離れたら、ケースを排斥テーブルの上へ置き直す（同じ物体を使い回す。製品テーブルの上は空になる）

使い方（リポジトリの直下で）:
  .venv\\Scripts\\python.exe src/fg/integrate.py --loco <run14 の .pt> --manip <V16 の .pt> --num-envs 64 [--trips 3] [--video out.mp4]
"""

import argparse
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("FG_FORCE_MODE", "0")
os.environ.setdefault("FG_EMPTY_FRACTION", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fg.pick_learn as pl  # noqa: E402
from fg.free_bin import layout  # noqa: E402
from fg.luna_constants import BIN_OUTER  # noqa: E402
from fg.pick_place import table_spec  # noqa: E402
from fg.terrains import PLATFORM_WIDTH, STEP_HEIGHT_RANGE, TILE_SIZE, TWO_STEP_PLATFORM, _BORDER  # noqa: E402
from mjlab.entity import EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply  # noqa: E402

H = 0.20  # 段の高さ
EDGE = TILE_SIZE / 2 - _BORDER  # 台の中心から 1 段目の縁 1.35 m
FLOOR = -2 * H
X_REJ = 0.55  # 排斥テーブル（台の上）
X_PROD = -(EDGE + 5.0)  # 製品テーブル（床）
OZ = BIN_OUTER[2] / 2
TOL = 0.04  # 寄るときの前後・横の許容 [m]（取り上げ・置くの学習は始めの位置を ±3 cm ばらつかせている）
VX = 0.4  # 歩く速さ（人の階段の速さ 0.45〜0.58 m/s より少し遅く。run9・run14 は 0.3〜0.58 で 100 %）

# 段階
(WALK_TO_REJ, FINE_REJ, PICK, BACK_REJ, TURN_REJ, WALK_TO_PROD, FINE_PROD, PLACE, BACK_PROD, TURN_PROD, DONE, FAIL) = range(12)
NAMES = ["排斥テーブルへ歩く（ステップを上る）", "排斥テーブルに寄る", "取り上げ", "下がる（取り上げた後）", "振り返る（台の上）",
         "製品テーブルへ歩く（ステップを降りる）", "製品テーブルに寄る", "置く", "下がる（置いた後）", "振り返る（製品テーブルの前）", "完了", "失敗"]


def wrap(a):
  return (a + math.pi) % (2 * math.pi) - math.pi


def make_env(num_envs, video, prod_top=None):
  cfg = load_env_cfg(pl.PICK_TASK_ID, play=True)
  cfg.scene.num_envs = num_envs
  lo, hi = STEP_HEIGHT_RANGE
  d = (H - lo) / (hi - lo)
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(TILE_SIZE, TILE_SIZE), border_width=8.0, num_rows=1, num_cols=1, curriculum=False,
    sub_terrains={"two_step": replace(TWO_STEP_PLATFORM, proportion=1.0)}, difficulty_range=(d, d))
  cfg.events.pop("randomize_terrain", None)
  # 製品テーブル（床）
  # 天板の左右の幅（既定 0.20 m は排斥テーブルと同じ。箱の底 0.254 m より狭く、置くときに箱が横へずれると天板から落ちた：2026-10-13 13:50）
  top_y = float(os.environ.get("FG_PROD_TOP_Y", "0.20"))
  cfg.scene.entities["table_b"] = EntityCfg(spec_fn=table_spec(pl.HB if prod_top is None else prod_top, "table_b", top_y=top_y), init_state=EntityCfg.InitialStateCfg(pos=(X_PROD, 0.0, FLOOR)))
  if cfg.sim.nconmax is not None:
    cfg.sim.nconmax += 20
  # スタート：製品テーブルの手前（置いて下がった所）、ステップの方を向く
  x0 = float(X_PROD + layout()["center_root"][0] + 0.35)
  cfg.events["reset_base"].params["pose_range"] = {"x": (x0, x0), "y": (-0.02, 0.02), "z": (FLOOR, FLOOR), "yaw": (0.0, 0.0)}
  # 学習用の終了・報酬（成功・落下で台が作り直されないように）
  for nm in ("task_success", "bin_dropped"):
    cfg.terminations.pop(nm, None)
  for nm in ("success", "manip_drop", "bin_drop"):
    cfg.rewards.pop(nm, None)
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.heading_command = False
  tw.ranges.heading = None
  tw.rel_standing_envs = 0.0
  tw.rel_heading_envs = 0.0
  tw.ranges.lin_vel_x = (0.0, 0.0)
  tw.ranges.lin_vel_y = (0.0, 0.0)
  tw.ranges.ang_vel_z = (0.0, 0.0)
  if video:
    cfg.viewer.env_idx = int(os.environ.get("FG_VIDEO_ENV", "0"))
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = (float(v) for v in os.environ.get("FG_CAM", "4.5,-18,120").split(","))
    cfg.viewer.width, cfg.viewer.height = 1280, 720
  return cfg


def load_policy(wenv, path):
  agent_cfg = load_rl_cfg(pl.PICK_TASK_ID)
  r = load_runner_cls(pl.PICK_TASK_ID)(wenv, asdict(agent_cfg), device="cuda:0")
  r.load(path, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return r.get_inference_policy(device="cuda:0")


def load_loco(path):
  """run14（入力 280 個）の方策を、学習の形のまま読む（正規化を含む）。"""
  sd = torch.load(path, map_location="cuda:0", weights_only=False)["actor_state_dict"]
  mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
  layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

  def f(x):
    x = (x[:, :280] - mean) / (std + 1e-2)
    for j, (w, b) in enumerate(layers):
      x = x @ w.T + b
      if j < 3:
        x = torch.nn.functional.elu(x)
    return x

  return f


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--loco", required=True, help="歩く・段差の方策（run14）")
  ap.add_argument("--manip", required=True, help="取り上げ・置くの方策（V16）")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--trips", type=int, default=3)
  ap.add_argument("--seconds", type=float, default=600.0)
  ap.add_argument("--video", type=str, default=None)
  ap.add_argument("--video-seconds", type=float, default=200.0)
  ap.add_argument("--replay-video", type=str, default=None, help="全台の姿勢を記録し、終わってから完了した 1 台を動画にする（25 fps）")
  ap.add_argument("--prod-top", type=float, default=None, help="製品テーブルの天板の高さ [m]（床から。既定は V16 の学習と同じ 938 mm）")
  ap.add_argument("--place-dz", type=float, default=-0.015, help="シーケンス制御で置くとき、抱える姿勢から下げる量 [m]")
  ap.add_argument("--manip-legs-loco", action="store_true", help="取り上げ・置くの間も脚は run14 に任せ、V16 は腕だけ")
  ap.add_argument("--flat-scan", action="store_true", help="取り上げ・置くの方策に、地形の高さの入力を平らに見せる")
  ap.add_argument("--place-scripted", action="store_true",
                  help="置くは学習した方策ではなく、シーケンス制御（run14 で立ち、腕を逆運動学で下ろして外へ開く。pick_place.py と同じやり方）")
  # 診断（2026-10-13 朝）の切り分け用
  ap.add_argument("--fresh-obs", action="store_true", help="モード・指令を決めた後に観測を作り直して方策へ渡す（切り替えの最初の 1 歩で古い観測を渡さない）")
  ap.add_argument("--reset-hist", action="store_true", help="取り上げ・置くに入るとき、前の行動（観測の last_action）を 0 にする（学習の始めと同じ）")
  ap.add_argument("--filt-actual", action="store_true", help="取り上げ・置くに入るとき、腕のフィルタを目標でなく実際の関節の角度で初期化する")
  ap.add_argument("--settle", type=float, default=0.0, help="寄り終えた後、胴体の速さ・角速度が小さくなるまで待つ（0 で無効。値は胴体の速さの上限 [m/s]）")
  ap.add_argument("--settle-correct", action="store_true", help="止まるのを待つ間も位置を直し続け、始める瞬間にも許容の内側を求める")
  ap.add_argument("--settle-max", type=float, default=2.0, help="止まりきるのを待つ最長の時間 [s]")
  ap.add_argument("--dead-frac", type=float, default=1.0, help="寄るとき、指令を出し続ける範囲（許容に対する割合。1 で従来どおり）")
  ap.add_argument("--tol-prod", type=float, default=TOL, help="製品テーブルに寄るときの前後・横の許容 [m]")
  ap.add_argument("--tol", type=float, default=TOL, help="寄るときの前後・横の許容 [m]")
  ap.add_argument("--place-closed", action="store_true",
                  help="置くのシーケンス制御を、時刻でなく確認で進める（載った・左右の重さが抜けた・両手が抜けた、を確かめてから次へ）。--place-scripted と一緒に使う")
  ap.add_argument("--place-v2", action="store_true",
                  help="置き方 v2：箱が載るまで下ろし、受け板をフランジと押さえの間（すき間 3 mm）に左右別に 2.5 mm 刻みで合わせ、確かめてから開く。抜けるまで上げない")
  ap.add_argument("--place-retry", type=int, default=0, help="確認で進む置き方で、抜けない手を 1 cm 下げて開き直す最大回数")
  ap.add_argument("--place-rel", type=float, default=0.0,
                  help="置く（--place-closed）で、測った時の箱（天板の上で止まった位置・向き）を基準に、今の上体の姿勢から腕の目標を直す割合（0 で無効、1 で全部）。"
                       "上体が箱に対して傾くと、上体の座標で決めた左右の手の高さがずれる、という仮説を確かめるために足した（2026-10-17）。"
                       "不採用：上体と箱の左右の傾きのずれは 1 度ほどで、右手が左手より高い原因ではなかった。1.0 にすると箱を落とす・左手が抜けない台が増えた")
  ap.add_argument("--place-low-dur", type=float, default=2.0, help="置く（--place-closed）で、抱える姿勢から下ろす時間 [s]（その後 0.4 秒待って測る）")
  ap.add_argument("--place-adj-max", type=float, default=4.0,
                  help="置く（--place-closed）で、合わせ始めてから、載ったと判定できなくても開くまでの時間 [s]。"
                       "Luna の完了した台の 99 % は、載った判定（gap<8 mm）に届かず、この 4 秒を待ってから開いていた（2026-10-17）")
  ap.add_argument("--place-max-sel", type=int, default=4, help="確認で進む置き方で、左右それぞれ追加で下げる最大 [cm]（0〜4）")
  ap.add_argument("--place-swap", action="store_true", help="確認で進める置き方で、左右の受け板と逆運動学の左右の対応を入れ替える")
  ap.add_argument("--prod-by-box", action="store_true", help="製品テーブルへは、抱えた箱の位置（胴体でなく）で寄せる")
  ap.add_argument("--pick-yaw", action="store_true", help="取り上げの成功に、向きのずれ 10 度未満・角速度 0.2 rad/s 未満を入れる")
  ap.add_argument("--start-place", type=str, default=None, help="調査用：entry-log の .pt の entry_place から、置くを始める")
  ap.add_argument("--lock-waist-place", choices=("pitch", "all"), default=None, help="置くの間、腰（pitch だけ、または 3 軸）を 0 に固定")
  ap.add_argument("--vx-down", type=float, default=VX, help="製品テーブルへ歩く（ステップを降りる）ときの最大の速さ [m/s]")
  ap.add_argument("--carry-dur", type=float, default=1.0, help="取り上げた後、腕を抱える姿勢へ移す時間 [s]")
  ap.add_argument("--back-stable", action="store_true", help="取り上げた後、腕・箱・体が落ち着いてから下がり、下がる間も向きを保つ")
  ap.add_argument("--place-hold", type=float, default=0.0,
                  help="置く（--place-closed）で腕を下ろす間、箱が天板に着くまで、置くに入った時の胴体の位置・向きを保つ指令を出す。値はこの距離 [m] 以上ずれたら動かす（0 で無効）")
  ap.add_argument("--fine-vx", type=float, default=0.12, help="寄る段階で前へ進む速さ [m/s]")
  ap.add_argument("--settle-qvel", type=float, default=0.0, help="取り上げを始める前に、関節の速さのノルムがこの値 [rad/s] 未満になるのも待つ（0 で無効）")
  ap.add_argument("--pick-hook-tol", type=float, default=0.02, help="取り上げの成功の、受け板のずれ（左右の平均）の許容 [m]")
  ap.add_argument("--carry-after-back", type=float, default=0.0, help="取り上げた後、この距離 [m] 下がってから抱える姿勢へ移る（0 で無効、--carry-delay より優先）")
  ap.add_argument("--carry-delay", type=float, default=0.0, help="取り上げた後、抱える姿勢へ移り始めるまでの時間 [s]（その間は持ち上げた姿勢のまま下がる）")
  ap.add_argument("--align-first", type=float, default=0.0, help="排斥テーブルに寄るとき、横にこの距離 [m] 以上ずれている間は 10 cm 手前までしか寄らない（0 で無効）")
  ap.add_argument("--align-back", type=float, default=0.10, help="--align-first で、横がずれている間に下がっておく距離 [m]")
  ap.add_argument("--fine-kick", type=float, default=0.0, help="排斥テーブルに寄り始めてこの時間 [s] たっても寄れない台は、指令を前 0.2・横 0.25 m/s に上げる（0 で無効）")
  ap.add_argument("--pick-retry", type=float, default=0.0, help="取り上げを始めてこの時間 [s] たっても箱が浮いていなければ、寄る段階へ戻ってやり直す（1 往復に 1 回、0 で無効）")
  ap.add_argument("--retry-wait", type=float, default=0.0, help="--pick-retry でやり直すとき、この時間 [s] は足を止めて手を開くだけにする")
  ap.add_argument("--step-wz-max", type=float, default=0.0, help="ステップを上る間の、向きを戻す指令の上限 [rad/s]（0 で無効）")
  ap.add_argument("--top-vy", type=float, default=0.0, help="台に上がった後、横のずれを戻す指令の上限 [m/s]（0 で今まで通り 0.15）")
  ap.add_argument("--step-wz-d", type=float, default=0.0, help="ステップを上る間、向きの補正に回る速さを打ち消す項を足す（係数、0 で無効）")
  ap.add_argument("--back-hold", action="store_true", help="置いた後に下がる間も、下がり始めた時の横の位置と向きを保つ")
  ap.add_argument("--yaw-first", type=float, default=0.0, help="排斥テーブルに寄るとき、向きがこの角度 [rad] 以上ずれている間は、その場で向きだけ直す（0 で無効）")
  ap.add_argument("--back-only-tol", action="store_true", help="排斥テーブルに寄るとき、近すぎても許容の内側なら下がらない")
  ap.add_argument("--step-novy", action="store_true", help="ステップを上る間は横の指令を 0 にする")
  ap.add_argument("--vx-up", type=float, default=VX, help="排斥テーブルへ歩く（ステップを上る）ときの最大の速さ [m/s]")
  ap.add_argument("--turn-ramp", type=float, default=0.0, help="置いた後に振り返るとき、回る指令をこの時間 [s] かけて上げる（0 で無効）")
  ap.add_argument("--turn-policy", type=str, default=None,
                  help="振り返る（製品テーブルの前、その場で回る）間だけ使う方策（回れ右の方策、src/fg/turn_skill.py。入力は run14 と同じ）")
  ap.add_argument("--turn-wmin", type=float, default=0.3, help="振り返るときの最小の角速度 [rad/s]（回れ右の方策は 0.3 では残り 15 度から 2 秒かかった：2026-10-17）")
  ap.add_argument("--turn-rej-loco-wmax", type=float, default=0.5, help="台の上の振り返りを歩行の方策でするとき（--turn-rej なし）の最大の角速度 [rad/s]")
  ap.add_argument("--turn-rej-wmax", type=float, default=1.0, help="--turn-rej の台の上の振り返り（箱を持つ）の最大の角速度 [rad/s]")
  ap.add_argument("--turn-rej-wmin", type=float, default=0.3, help="--turn-rej の台の上の振り返りの最小の角速度 [rad/s]")
  ap.add_argument("--climb-policy", type=str, default=None,
                  help="排斥テーブルへ歩く（ステップを上る）間だけ使う歩行の方策（入力は run14 と同じ。run30 は上りの転倒が少ないが寄る・下がるが崩れた：2026-10-17）")
  ap.add_argument("--climb-desc", action="store_true", help="--climb-policy を、製品テーブルへ歩く（ステップを降りる）間にも使う")
  ap.add_argument("--turn-rej", action="store_true", help="--turn-policy を、台の上の振り返り（取り上げた後）にも使う")
  ap.add_argument("--turn-wmax", type=float, default=0.5, help="置いた後に振り返るときの最大の角速度 [rad/s]")
  ap.add_argument("--open-dur", type=float, default=1.0, help="置くで、受け具を外へ開く時間 [s]（シーケンス制御）")
  ap.add_argument("--up-spread", type=float, default=0.0, help="置くで、開いた後に上げる姿勢を、さらにこの距離 [m] 外へ開く")
  ap.add_argument("--open-hold", action="store_true", help="置くで、開き始めてから終わるまで、開き始めた時の胴体の位置・向きを保つ指令を出す（--place-hold の距離を使う）")
  ap.add_argument("--place-hold-ph1", action="store_true", help="--place-hold を、測って合わせる間（開く前まで）も続ける")
  ap.add_argument("--place-hold-box", action="store_true", help="--place-hold で、胴体の位置ではなく箱をテーブルの中心へ寄せる")
  ap.add_argument("--pick-dxy", type=float, default=0.0, help="取り上げの成功に、箱がテーブルの中心からこの距離 [m] 以内を入れる（0 で無効）")
  ap.add_argument("--back-wait", type=float, default=0.5, help="取り上げた後、下がり始めるまで待つ時間 [s]")
  ap.add_argument("--back-prod", type=float, default=0.3, help="置いた後に下がる距離 [m]")
  ap.add_argument("--seed", type=int, default=None, help="シミュレーションの乱数の種")
  ap.add_argument("--metrics", type=str, default=None,
                  help="所要時間と滑らかさを段階ごとに集めて、完了した台の平均を表で出し、.json に保存する（2026-10-17）")
  ap.add_argument("--residual-train", type=str, default=None,
                  help="置くの残差強化学習（src/fg/place_residual.py）を行い、このフォルダに保存する。--start-place の記録から全台を置くで始め、"
                       "置く → 下がるを 1 回の試行にして繰り返す（2026-10-18）")
  ap.add_argument("--residual-policy", type=str, default=None, help="学習した置くの残差の方策（.pt）を、置く・下がる間に使う")
  ap.add_argument("--residual-scale", type=float, default=0.04, help="腕の補正の最大 [rad]")
  ap.add_argument("--residual-no-open", action="store_true", help="開く合図を使わない（腕の補正だけ）")
  ap.add_argument("--residual-iters", type=int, default=200, help="学習の回数")
  ap.add_argument("--residual-T", type=int, default=24, help="1 回の学習で集める判断の数（0.1 秒ごと）")
  ap.add_argument("--residual-time-w", type=float, default=1.0, help="時間の罰（1 秒あたり）")
  ap.add_argument("--residual-fail-w", type=float, default=10.0,
                  help="失敗の罰（res1 は 10 で、早く開いて 4 秒縮める代わりに成功が約 99 → 97.5 % に下がった：2026-10-18）")
  ap.add_argument("--entry-log", type=str, default=None, help="取り上げに入った瞬間の状態を各台について保存（.pt）")
  a = ap.parse_args()
  dev = "cuda:0"
  cfg = make_env(a.num_envs, a.video or a.replay_video, a.prod_top)
  if a.residual_train:
    cfg.episode_length_s = 1e7  # 学習は長く回すので、環境の時間切れで台が作り直されないように
  if a.seed is not None:
    cfg.seed = a.seed
    torch.manual_seed(a.seed)
  env = ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array" if (a.video or a.replay_video) else None)
  qhist = []
  wenv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(pl.PICK_TASK_ID).clip_actions)
  manip = load_policy(wenv, a.manip)
  loco = load_loco(a.loco)
  turn_pol = load_loco(a.turn_policy) if a.turn_policy else None
  climb_pol = load_loco(a.climb_policy) if a.climb_policy else None
  env.reset()
  n = a.num_envs
  org = env.scene.env_origins
  robot, box = env.scene["robot"], env.scene["bin"]
  at = env.action_manager.get_term("joint_pos")
  cmd = env.command_manager.get_term("twist")
  at.schedule = False
  carry_q, ready_q, raise_q = pl._arm_q(env)
  if a.place_scripted:
    from fg.arm_ik import ArmIK

    ik = ArmIK()

    def q_of(**kw):
      r, _, _ = ik.solve(**kw)
      return torch.tensor([r[nm] for nm in at.arm_names], device=dev)

    # 下げる量は 1.5 cm（0.5 cm では腕のたわみで箱が 5〜7 mm 浮いたまま片手に掛かり、開くと箱を引きずった：2026-10-13 3:10）
    # 開くときに 3 cm さらに下げると、箱が傾いて落下が 1 → 19 台に増えた（2026-10-13 3:45）ので、同じ高さで開く
    low_q, open_q, openup_q = q_of(dz=a.place_dz), q_of(dz=a.place_dz, spread=pl.SPREAD), q_of(dz=a.place_dz + 0.04, spread=pl.SPREAD)
    # 左右別の追加の下げ（0〜4 cm、1 cm 刻み）の姿勢の表：載せた後、左右の受け板の高さを測り、高い方だけ余分に下げてから開く
    # （置くの失敗の多くは、左右の手の高さの差で片方の受け板に箱の重さが残り、開くと箱を引きずっていた：2026-10-13 3:40）
    G = [0.0, -0.01, -0.02, -0.03, -0.04]
    low2_tab = torch.zeros(5, 5, len(at.arm_names), device=dev)
    open2_tab, up2_tab = low2_tab.clone(), low2_tab.clone()
    for i_, gl in enumerate(G):
      for j_, gr in enumerate(G):
        so = {"left": [0.0, 0.0, gl], "right": [0.0, 0.0, gr]}
        low2_tab[i_, j_] = q_of(dz=a.place_dz, side_off=so)
        open2_tab[i_, j_] = q_of(dz=a.place_dz, spread=pl.SPREAD, side_off=so)
        up2_tab[i_, j_] = q_of(dz=a.place_dz + 0.04, spread=pl.SPREAD + a.up_spread, side_off=so)
    sel_l = torch.zeros(n, dtype=torch.long, device=dev)
    sel_r = torch.zeros(n, dtype=torch.long, device=dev)
    decided = torch.zeros(n, dtype=torch.bool, device=dev)
    # 箱の上体に対するずれ δ（位置 3・回転 3、上体の座標）→ 腕の関節角の変化（1 次）。下ろした構えと開いた構えの 2 つ
    J_low = torch.tensor(ik.rel_jacobian(list(at.arm_names), dz=a.place_dz), dtype=torch.float32, device=dev)
    J_open = torch.tensor(ik.rel_jacobian(list(at.arm_names), dz=a.place_dz, spread=pl.SPREAD), dtype=torch.float32, device=dev)
    R0_t = torch.tensor(ik.Rt.T, dtype=torch.float32, device=dev)  # 抱える姿勢での箱の向き（上体の座標。箱は世界の向き）
    bin_t0 = torch.tensor(ik.bin_t, dtype=torch.float32, device=dev)  # 抱える姿勢での箱の中心（上体の座標）
    if a.place_v2:
      # 片側の上下の補正 +1 cm 〜 −5 cm、2.5 mm 刻み（25 段）。押さえのすき間は 3 mm なので、1 cm 刻みでは合わせられない
      # 逆運動学の腕（肩・肘）は、手を下げると手の前後の傾きが 1 cm あたり約 4 度変わる（−2 cm で +6.6 度）。受け板がフランジの下で
      # 傾くと、端が箱を持ち上げる（v2 の 2.5 mm 刻み・−5 cm までの版は、下げるほど箱が浮いた：2026-10-13 9:20）→ 1 mm 刻み、+1〜−1.5 cm
      GV = [0.01 - 0.001 * k for k in range(26)]
      K0 = 10  # 補正 0 の段
      is_left = torch.tensor(["left" in nm for nm in at.arm_names], device=dev)
      v2_tab = {}
      for nm_, kw_ in (("low", dict(dz=a.place_dz)), ("open", dict(dz=a.place_dz, spread=pl.SPREAD)), ("up", dict(dz=a.place_dz + 0.04, spread=pl.SPREAD))):
        tl = torch.stack([q_of(**kw_, side_off={"left": [0.0, 0.0, g], "right": [0.0, 0.0, 0.0]}) for g in GV])
        tr = torch.stack([q_of(**kw_, side_off={"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, g]}) for g in GV])
        v2_tab[nm_] = (tl, tr)

      def v2_q(nm_, kl, kr):
        tl, tr = v2_tab[nm_]
        return torch.where(is_left[None, :], tl[kl], tr[kr])

      kL = torch.full((n,), K0, dtype=torch.long, device=dev)
      kR = torch.full((n,), K0, dtype=torch.long, device=dev)
      v2_ok = torch.zeros(n, device=dev)
  # 確認で進める置き方の段階：0 下ろす、1 合わせる（左右を 1 cm ずつ下げて測り直す）、2 開く、3 上げる、4 終わり
  ph = torch.zeros(n, dtype=torch.long, device=dev)
  is_l_arm = torch.tensor(["left" in nm for nm in at.arm_names], device=dev)
  torso_id = robot.find_bodies(("waist_pitch_link",))[0][0]
  elbow_l = next(j for j, nm in enumerate(at.arm_names) if "left" in nm and "elbow" in nm)
  ph_t = torch.zeros(n, device=dev)
  n_retry = torch.zeros(n, dtype=torch.long, device=dev)
  ph_started = torch.zeros(n, dtype=torch.bool, device=dev)
  # --place-rel：測った時の箱の位置・向き（区画の座標）、使うか、今足している腕の補正
  rel_bp = torch.zeros(n, 3, device=dev)
  rel_bq = torch.zeros(n, 4, device=dev)
  rel_on = torch.zeros(n, dtype=torch.bool, device=dev)
  rel_add = torch.zeros(n, len(at.arm_names), device=dev)

  def _rot(q):
    """四元数（w, x, y, z）→ 回転行列（台数 × 3 × 3）"""
    e = torch.eye(3, device=dev)
    return torch.stack([quat_apply(q, e[k].expand(len(q), 3)) for k in range(3)], dim=-1)

  def rel_delta(bp_, bq_):
    """箱（bp_・bq_、区画の座標）の、今の上体に対する、抱える姿勢からのずれ δ（位置 3 [m]・回転ベクトル 3 [rad]、上体の座標）"""
    pT = robot.data.body_link_pos_w[:, torso_id] - org
    RT = _rot(robot.data.body_link_quat_w[:, torso_id])
    d = torch.einsum("nji,nj->ni", RT, bp_ - pT) - bin_t0
    dR = RT.transpose(1, 2) @ _rot(bq_) @ R0_t.T
    v = 0.5 * torch.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]], dim=-1)
    return torch.cat([d.clamp(-0.05, 0.05), v.clamp(-0.2, 0.2)], dim=-1)
  place_trace = [None] * n
  pick_trace = [None] * n
  appr_trace = [[] for _ in range(n)]
  # 置くに入った時の胴体の位置・向き（--place-hold で、箱が天板に着くまで保つ）
  hold_xy = torch.zeros(n, 2, device=dev)
  hold_yaw = torch.zeros(n, device=dev)
  open_xy = torch.zeros(n, 2, device=dev)
  open_yaw = torch.zeros(n, device=dev)
  c_root =torch.tensor(layout()["center_root"], dtype=torch.float32, device=dev)
  # テーブル（mocap）：排斥テーブルは台の上、製品テーブルは床
  dz_a = torch.zeros(n, device=dev)
  all_ids = torch.arange(n, device=dev)
  tq = torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)
  env.scene["table"].write_mocap_pose_to_sim(torch.cat([org + torch.tensor([X_REJ, 0.0, 0.0], device=dev) + torch.stack([0 * dz_a, 0 * dz_a, dz_a], -1), tq], -1), env_ids=all_ids)
  env.scene["table_b"].write_mocap_pose_to_sim(torch.cat([org + torch.tensor([X_PROD, 0.0, FLOOR], device=dev), tq], -1), env_ids=all_ids)
  top_rej = pl.HA
  top_prod = FLOOR + (pl.HB if a.prod_top is None else a.prod_top)

  def spawn_box(mask):
    ids = torch.nonzero(mask).flatten()
    if len(ids) == 0:
      return
    p = org[ids] + torch.tensor([X_REJ, 0.0, top_rej + OZ + 0.002], device=dev)
    box.write_root_link_pose_to_sim(torch.cat([p, tq[ids]], -1), env_ids=ids)
    box.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)

  spawn_box(torch.ones(n, dtype=torch.bool, device=dev))
  # 腕：手ぶらは受け具を構えた姿勢、運ぶときは抱える姿勢（なめらかに移す）
  at.arm_target = ready_q.expand(n, -1).clone()
  at.arm_target_mask = torch.ones(n, dtype=torch.bool, device=dev)
  arm_from, arm_to = at.arm_target.clone(), at.arm_target.clone()
  arm_t0 = torch.zeros(n, device=dev)

  def set_arm(mask, q, t, dur=1.0):
    arm_from[mask] = at.arm_target[mask] - rel_add[mask] - res_add[mask]
    arm_to[mask] = q
    arm_t0[mask] = t
    arm_dur[mask] = dur

  arm_dur = torch.ones(n, device=dev)
  stage = torch.full((n,), WALK_TO_REJ, dtype=torch.long, device=dev)
  t_stage = torch.zeros(n, device=dev)
  trips = torch.zeros(n, dtype=torch.long, device=dev)
  hold = torch.zeros(n, device=dev)
  arrived = torch.zeros(n, dtype=torch.bool, device=dev)
  n_pick_retry = torch.zeros(n, dtype=torch.long, device=dev)
  back_yaw = torch.zeros(n, device=dev)
  back_from = torch.zeros(n, 2, device=dev)
  carry_pending = torch.zeros(n, dtype=torch.bool, device=dev)
  fail_stage = torch.full((n,), -1, dtype=torch.long, device=dev)
  fail_why = [""] * n
  entry = [None] * n
  entry_place = [None] * n
  entry_place_all = []
  t_done = torch.full((n,), float("nan"), device=dev)
  t_fail = torch.full((n,), float("nan"), device=dev)
  pl.task_mode(env)
  env._fg_arm_base = ready_q.expand(n, -1).clone()
  env._fg_seat_t = torch.zeros(n, device=dev)
  if not hasattr(at, "_arm_filt"):
    at._arm_filt = ready_q.expand(n, -1).clone()
  env._fg_table_xy = torch.tensor([X_REJ, 0.0], device=dev).expand(n, 2).clone()
  env._fg_table_top[:] = top_rej
  wrist_ids = robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"))[0]
  Lo = layout()
  off_mid = torch.tensor(Lo["center_w"] - (Lo["data"].xpos[Lo["lw"]] + Lo["data"].xpos[Lo["rw"]]) / 2, dtype=torch.float32, device=dev)
  t_ep0 = torch.zeros(n, device=dev)  # 置くの残差の学習：試行を始めた時刻
  if a.start_place:
    # 調査用：統合で記録した「置くに入った瞬間」から、いきなり置くを始める（学習した置くの単体と統合の食いちがいの切り分け：2026-10-13 16:20）
    pb_ = torch.load(a.start_place, weights_only=False)
    pb = pb_.get("entry_place_all") or pb_["entry_place"]
    pb = [e for e in pb if e is not None]
    print(f"置くの始めの記録：{len(pb)} 件", flush=True)
    jj = torch.randint(0, len(pb), (n,)).tolist()
    c0 = torch.tensor([X_PROD, 0.0, FLOOR], device=dev)
    T = lambda k: torch.tensor([pb[j][k] for j in jj], dtype=torch.float32, device=dev)  # noqa: E731
    robot.write_root_link_pose_to_sim(torch.cat([org + c0 + T("root_pos"), T("root_quat")], -1), env_ids=all_ids)
    robot.write_root_link_velocity_to_sim(torch.cat([T("root_lin_w"), T("root_ang_w")], -1), env_ids=all_ids)
    robot.write_joint_state_to_sim(T("q"), T("qvel"), env_ids=all_ids)
    box.write_root_link_pose_to_sim(torch.cat([org + c0 + T("box_pos"), T("box_quat")], -1), env_ids=all_ids)
    box.write_root_link_velocity_to_sim(torch.cat([T("box_lin_w"), T("box_ang_w")], -1), env_ids=all_ids)
    stage[:] = PLACE
    env._fg_table_xy[:] = torch.tensor([X_PROD, 0.0], device=dev)
    env._fg_table_top[:] = top_prod
    env._fg_arm_base[:] = raise_q
    env._fg_table_yaw = torch.full((n,), math.pi, device=dev)
    env._fg_seated[:] = False
    at._arm_filt[:] = T("q")[:, env._fg_arm_jids]
    at.arm_target[:] = T("q")[:, env._fg_arm_jids]
    arm_from[:] = at.arm_target
    arm_to[:] = at.arm_target
    am_ = env.action_manager
    am_._action[:] = 0.0
    am_._prev_action[:] = 0.0
    at._raw_actions[:] = 0.0
    # 書き込んだ状態から体の位置・向きを計算し直す（しないと最初の 1 歩で古い向き 0 を読み、置くの位置保持が向き 0 へ回そうとした：2026-10-18）
    env.sim.forward()
    env.observation_manager._obs_buffer = None
  obs = wenv.get_observations()
  dt = env.step_dt
  frames, log = [], []
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  steps = int(a.seconds / dt)
  # --- 置くの残差（2026-10-18）：腕の補正 res_add と開く合図 res_open。0.1 秒（5 歩）ごとに決める
  res_on = bool(a.residual_train or a.residual_policy)
  res_add = torch.zeros(n, len(at.arm_names), device=dev)
  res_open = torch.zeros(n, dtype=torch.bool, device=dev)
  RES_K = 5
  if res_on:
    import json
    import time as _time
    from collections import Counter as _Counter

    from fg import place_residual as PR

    assert a.place_closed, "置くの残差は --place-closed の手順の上に足す"
    rac = PR.ResidualAC().to(dev)
    if a.residual_policy:
      rac.load_state_dict(torch.load(a.residual_policy, map_location=dev, weights_only=False)["model"])
    if a.residual_train:
      assert a.start_place, "--residual-train には --start-place（置くに入った瞬間の記録）が要る"
      os.makedirs(a.residual_train, exist_ok=True)
      ppo = PR.PPO(rac)
      steps = a.residual_iters * a.residual_T * RES_K + 1
      res_prev = None  # 前の判断の（入力・行動・確率・価値）
      res_stat = {"成功": 0, "失敗": 0, "成功の時間": 0.0, "早く開いた": 0}
      res_log, res_best, res_t0 = [], -1.0, _time.time()
      ep_open = torch.zeros(n, dtype=torch.bool, device=dev)
      res_fail = []

      def place_reset(ids):
        """台 ids を、記録からランダムに選んだ「置くに入った瞬間」へ戻す"""
        jj = torch.randint(0, len(pb), (len(ids),)).tolist()
        c0 = torch.tensor([X_PROD, 0.0, FLOOR], device=dev)
        T_ = lambda k_: torch.tensor([pb[j][k_] for j in jj], dtype=torch.float32, device=dev)  # noqa: E731
        robot.write_root_link_pose_to_sim(torch.cat([org[ids] + c0 + T_("root_pos"), T_("root_quat")], -1), env_ids=ids)
        robot.write_root_link_velocity_to_sim(torch.cat([T_("root_lin_w"), T_("root_ang_w")], -1), env_ids=ids)
        robot.write_joint_state_to_sim(T_("q"), T_("qvel"), env_ids=ids)
        box.write_root_link_pose_to_sim(torch.cat([org[ids] + c0 + T_("box_pos"), T_("box_quat")], -1), env_ids=ids)
        box.write_root_link_velocity_to_sim(torch.cat([T_("box_lin_w"), T_("box_ang_w")], -1), env_ids=ids)
        stage[ids] = PLACE
        t_stage[ids] = t
        hold[ids] = 0.0
        ph_started[ids] = False
        fell[ids] = False
        t_fail[ids] = float("nan")
        env._fg_seated[ids] = False
        qa_ = T_("q")[:, env._fg_arm_jids]
        at._arm_filt[ids] = qa_
        at.arm_target[ids] = qa_
        arm_from[ids] = qa_
        arm_to[ids] = qa_
        rel_add[ids] = 0.0
        res_add[ids] = 0.0
        res_open[ids] = False
        ep_open[ids] = False
        am_ = env.action_manager
        am_._action[ids] = 0.0
        am_._prev_action[ids] = 0.0
        at._raw_actions[ids] = 0.0
        for i_ in ids.tolist():
          place_trace[i_] = None
          fail_why[i_] = ""

  def res_obs():
    """置くの残差の方策の入力（今の状態から）"""
    s_ = pl.task_state(env)
    p_ = robot.data.root_link_pos_w - org
    yaw_ = pl._yaw(robot.data.root_link_quat_w)
    c2, s2 = torch.cos(yaw_), torch.sin(yaw_)
    dtab = torch.tensor([X_PROD, 0.0], device=dev) - p_[:, :2]
    body_tab = torch.stack([c2 * dtab[:, 0] + s2 * dtab[:, 1], -s2 * dtab[:, 0] + c2 * dtab[:, 1]], -1)
    box_tab = s_["bp"][:, :2] - torch.tensor([X_PROD, 0.0], device=dev)
    jq = robot.data.joint_pos[:, env._fg_arm_jids]
    jv = robot.data.joint_vel[:, env._fg_arm_jids]
    return PR.build_obs(ph, stage == BACK_PROD, t - t_stage, t - ph_t, s_["hook_err"], s_["gap"], torch.linalg.norm(s_["bv"], dim=-1), s_["tilt"], s_["dxy"],
                        rel_delta(s_["bp"], s_["bq"]), jq - (at.arm_target - res_add), jv,
                        robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b, robot.data.projected_gravity_b,
                        body_tab, box_tab, torch.stack([sel_l, sel_r], -1))

  for k in range(steps):
    t = k * dt
    if res_on and k % RES_K == 0:
      if a.residual_train and res_prev is not None:
        # 前の判断からの 0.1 秒の報酬：時間の罰、腕の補正の大きさの罰、箱の傾きの罰。終わった台は成功 +10・失敗 −10
        s_now = pl.task_state(env)
        ended = (stage != PLACE) & (stage != BACK_PROD)
        succ = ended & (stage == TURN_PROD)
        fail_ = ended & ~succ
        o_p, a_p, lp_p, v_p = res_prev
        r_ = -a.residual_time_w * RES_K * dt - 0.05 * a_p[:, :PR.N_ARM].clamp(-1, 1).square().mean(-1) - 0.5 * s_now["tilt"] * RES_K * dt
        r_ = r_ + 10.0 * succ.float() - a.residual_fail_w * fail_.float()
        ppo.add(o_p, a_p, lp_p, v_p, r_, ended.float())
        res_stat["成功"] += int(succ.sum())
        res_stat["失敗"] += int(fail_.sum())
        res_stat["成功の時間"] += float((t - t_ep0)[succ].sum())
        res_stat["早く開いた"] += int((succ & ep_open).sum())
        res_fail += [fail_why[i_] for i_ in torch.nonzero(fail_).flatten().tolist()]
        if ended.any():
          ids_ = torch.nonzero(ended).flatten()
          place_reset(ids_)
          t_ep0[ids_] = t
          env.sim.forward()
          env.observation_manager._obs_buffer = None
          obs = wenv.get_observations()
      o_now = res_obs()
      if a.residual_train:
        with torch.no_grad():
          d_ = rac.dist(o_now)
          a_now = d_.sample()
          lp_now = d_.log_prob(a_now).sum(-1)
          v_now = rac.value(o_now)
        if len(ppo.buf) >= a.residual_T:
          st_ = ppo.update(v_now)
          it_ = len(res_log) + 1
          ne_ = res_stat["成功"] + res_stat["失敗"]
          sr_ = res_stat["成功"] / max(ne_, 1)
          rec_ = {"iter": it_, "episodes": ne_, "success": sr_, "succ_time": res_stat["成功の時間"] / max(res_stat["成功"], 1),
                  "early_open": res_stat["早く開いた"] / max(res_stat["成功"], 1), "std": rac.log_std.exp().tolist(),
                  "open_bias": float(rac.actor[-1].bias[PR.N_ARM]), **st_, "wall": round(_time.time() - res_t0),
                  "fail": dict(_Counter(w.split("：")[-1] for w in res_fail))}
          res_log.append(rec_)
          print(f"[残差] {it_} 回 試行 {ne_} 成功 {sr_ * 100:.1f} % 成功の時間 {rec_['succ_time']:.2f} 秒 早く開いた {rec_['early_open'] * 100:.0f} % "
                f"方策 {st_['pi']:+.3f} 価値 {st_['vf']:.3f} 腕の std {rec_['std'][0]:.3f} 開く std {rec_['std'][-1]:.2f} {rec_['wall']} 秒 "
                f"失敗 {dict(_Counter(w.split('：')[-1] for w in res_fail).most_common(3))}", flush=True)
          res_stat = {"成功": 0, "失敗": 0, "成功の時間": 0.0, "早く開いた": 0}
          res_fail = []
          sv_ = {"model": rac.state_dict(), "iter": it_, "args": vars(a)}
          torch.save(sv_, os.path.join(a.residual_train, "last.pt"))
          if it_ % 25 == 0:
            torch.save(sv_, os.path.join(a.residual_train, f"model_{it_}.pt"))
          with open(os.path.join(a.residual_train, "log.json"), "w", encoding="utf-8") as f_:
            json.dump(res_log, f_, ensure_ascii=False)
        res_prev = (o_now, a_now, lp_now, v_now)
      else:
        a_now = rac.act_mean(o_now)
      act_on = (stage == PLACE) | (stage == BACK_PROD)
      res_add = torch.where(act_on[:, None], a.residual_scale * a_now[:, :PR.N_ARM].clamp(-1, 1), torch.zeros_like(res_add))
      res_open = act_on & (a_now[:, PR.N_ARM] > 0) & (not a.residual_no_open)
      if a.residual_train:
        ep_open |= res_open & (stage == PLACE) & (ph == 1)
    p = robot.data.root_link_pos_w - org
    yaw = pl._yaw(robot.data.root_link_quat_w)
    s = pl.task_state(env)
    mid = robot.data.body_link_pos_w[:, wrist_ids].mean(dim=1) - org
    c_, s_ = torch.cos(yaw), torch.sin(yaw)
    pocket = mid[:, :2] + torch.stack([c_ * off_mid[0] - s_ * off_mid[1], s_ * off_mid[0] + c_ * off_mid[1]], -1)
    vx = torch.zeros(n, device=dev)
    vy = torch.zeros(n, device=dev)
    wz = torch.zeros(n, device=dev)
    mode = torch.zeros(n, dtype=torch.long, device=dev)

    def head(target_yaw, mask):
      e = wrap(target_yaw - yaw)
      w = torch.where(e.abs() > 0.03, torch.sign(e) * torch.clamp(1.0 * e.abs(), 0.2, 0.5), torch.zeros_like(e))
      return torch.where(mask, w, wz)

    # --- 歩く（向き 0 で +x へ、または向き π で −x へ）。y=0 の線に沿わせる
    for st, tx, tyaw in ((WALK_TO_REJ, X_REJ - c_root[0] - 0.35, 0.0), (WALK_TO_PROD, X_PROD + c_root[0] + 0.35, math.pi)):
      m = stage == st
      if m.any():
        sgn = 1.0 if tyaw == 0.0 else -1.0
        rem = sgn * (tx - p[:, 0])
        vmax = a.vx_down if st == WALK_TO_PROD else a.vx_up
        vx = torch.where(m, torch.clamp(0.8 * rem, 0.15, vmax), vx)
        vy = torch.where(m, torch.clamp(-sgn * 1.0 * p[:, 1], -0.15, 0.15), vy)
        if st == WALK_TO_REJ and a.step_novy:
          # ステップを上る間（1 段目の縁の 30 cm 手前から台の上 50 cm まで）は横の指令を 0 にする。上りで転ぶ台は、段の上で
          # 右へ 10〜30 度回りながら横へ ±10〜30 cm 振れていた（2026-10-16、appr_trace）
          on_step = (p[:, 0] > -EDGE - 0.3) & (p[:, 0] < -EDGE + 0.85)
          vy = torch.where(m & on_step, 0.0, vy)
        if st == WALK_TO_REJ and a.top_vy > 0:
          # 台に上がった後（1 段目の縁から 85 cm より先）は、横のずれを速く戻す（最大 a.top_vy m/s）。歩き終えた時点で横に 20 cm 超
          # ずれた台が 1019 台中 28 台あり、寄る段階で待たされたり、箱を押し落としたりしていた（2026-10-16）
          on_top = p[:, 0] >= -EDGE + 0.85
          vy = torch.where(m & on_top, torch.clamp(-sgn * 1.5 * p[:, 1], -a.top_vy, a.top_vy), vy)
        wz = head(torch.full((n,), tyaw, device=dev), m)
        if st == WALK_TO_REJ and a.step_wz_d > 0:
          # 段の上では、向きのずれに加えて回る速さも打ち消す（上限 0.5 rad/s は同じ）。転ぶ台は段の上で右へ回り続けていた。
          # 向きを戻す指令を小さくする案（--step-wz-max）は悪化したので、弱めずに回り続けるのを早く止める（2026-10-16 午後）
          on_step3 = (p[:, 0] > -EDGE - 0.3) & (p[:, 0] < -EDGE + 0.85)
          e3 = wrap(tyaw - yaw)
          w3 = torch.clamp(1.0 * e3 - a.step_wz_d * robot.data.root_link_ang_vel_w[:, 2], -0.5, 0.5)
          wz = torch.where(m & on_step3, w3, wz)
        if st == WALK_TO_REJ and a.step_wz_max > 0:
          # 段の上では向きを戻す指令も小さくする（転ぶ台は段の上で 10〜30 度回り、向きを戻す 0.5 rad/s の指令が出ていた）
          on_step2 = (p[:, 0] > -EDGE - 0.3) & (p[:, 0] < -EDGE + 0.85)
          wz = torch.where(m & on_step2, torch.clamp(wz, -a.step_wz_max, a.step_wz_max), wz)
        nx = m & (rem < 0.02)
        if st == WALK_TO_PROD:
          # 製品テーブルの 1 m 手前で、箱を 3 cm 上げる（置き方は V16 の置く課題の始めの姿勢と同じ）
          up = m & (rem < 1.0) & (arm_to - raise_q).abs().amax(-1).gt(1e-4)
          set_arm(up, raise_q, t, 1.0)
        stage[nx] = st + 1
        t_stage[nx] = t
        hold[nx] = 0.0
    # --- 寄る（受け具の中心を箱／テーブルの中心へ。前後 3 cm・横 3 cm・向き 2 度で 0.5 秒）
    for st, gx, tyaw in ((FINE_REJ, None, 0.0), (FINE_PROD, X_PROD, math.pi)):
      m = stage == st
      if m.any():
        goal = s["bp"][:, :2] if gx is None else torch.tensor([gx, 0.0], device=dev).expand(n, 2)
        # 胴体の目標＝箱が抱える位置に来る所（学習の始めの場面と同じ決め方）。受け具の中心で寄せると、手を開いた構えでは
        # 手首の位置が抱える姿勢とちがい、胴体が 11 cm テーブルに近づきすぎて取り上げの方策が転んだ（2026-10-13 2:30）
        cy_, sy_ = math.cos(tyaw), math.sin(tyaw)
        root_goal = goal - torch.tensor([cy_ * float(c_root[0]) - sy_ * float(c_root[1]), sy_ * float(c_root[0]) + cy_ * float(c_root[1])], device=dev)
        # 少し手前（3 cm）を狙う：学習の始めの場面は「目の前 ±3 cm」と「0.02〜0.15 m 手前」なので、テーブルに近すぎる側（前へ 3〜4 cm）は
        # 学習に無い。統合で取り上げに転ぶ台は、前後 −3.9 cm（近すぎ）で寄り終えていた（2026-10-13 3:50）
        # 製品テーブル（置く）には使わない：置く場所が 3 cm 手前にずれ、許容 4 cm と合わせて箱がテーブルの中心から 8 cm 超外れていた（2026-10-13 7:56）
        back = float(os.environ.get("FG_APPROACH_BACK", "0.03")) if st == FINE_REJ else float(os.environ.get("FG_APPROACH_BACK_PROD", "0.0"))
        root_goal = root_goal - back * torch.tensor([cy_, sy_], device=dev)
        e = root_goal - p[:, :2]
        tol_ = a.tol if st == FINE_REJ else a.tol_prod
        # 動かす指令は許容の半分まで出し続ける（許容ちょうどで指令を止めると、許容の縁 ±4.0 cm で行き来して 0.5 秒保てず、
        # 寄れない台が 1〜2 割あった：2026-10-13 9:08）
        dead_ = tol_ * a.dead_frac
        if st == FINE_PROD and a.prod_by_box:
          # 製品テーブルへは、抱えた箱そのものの位置で寄せる。運ぶ間に箱が受け具の上で前後・左右に最大 14 cm ずれていて、
          # 胴体の位置で寄せると箱がテーブルの中心から外れた（2026-10-13 8:06、置くの「中心から外れた」29 台）
          e = goal - s["bp"][:, :2]
        ex = c_ * e[:, 0] + s_ * e[:, 1]
        ey = -s_ * e[:, 0] + c_ * e[:, 1]
        # 後ろ向きは 0.2 m/s（0.15 では方策が足を出さず、前後 −3.1 cm で止まったまま 30 秒：2026-10-13 3:20）
        # 前へ寄る速さ（既定 0.12 m/s）。排斥テーブルの 10 cm 手前で止まったまま 30 秒寄れない台があった（小さな指令に足を出さない）：2026-10-16
        ex_c = ex
        if st == FINE_REJ and a.align_first > 0:
          # 横に a.align_first m 以上ずれている間は、テーブルの 10 cm 手前までしか寄らない（横を先に合わせる（2026-10-16）。
          # 寄れない台は、前後は合っているのに横に 11〜45 cm ずれたまま 30 秒止まっていた
          # 10 cm では足りなかった：取り上げの前に箱が落ちる台は、歩き終えた時点で横に 40〜50 cm ずれていて、斜めに寄る間に
          # 右の受け具が箱を横から押し、天板から +y へ落としていた（2026-10-16、appr_trace）。下がる距離は --align-back
          ex_c = torch.where(ey.abs() > a.align_first, ex - a.align_back, ex)
        fvx, fvy = a.fine_vx, 0.2
        if st == FINE_REJ and a.fine_kick > 0:
          # 寄り始めて a.fine_kick 秒たっても寄れない台は、指令を大きくする（小さな指令に足を出さず、10〜19 cm 手前で止まっていた台があった）
          kick = (t - t_stage) > a.fine_kick
          fvx = torch.where(kick, 0.2, a.fine_vx)
          fvy = torch.where(kick, 0.25, 0.2)
        vx = torch.where(m, torch.where(ex_c.abs() > dead_, torch.where(ex_c > 0, fvx, -0.2), 0.0), vx)
        if st == FINE_REJ and a.back_only_tol:
          # 行き過ぎ（近すぎ）が許容の内側なら下がらない。歩いてきた勢いで 2 cm ほど行き過ぎた台が −0.2 m/s で急に下がり、
          # 腕が振れて箱を押し落としていた（2026-10-16 夕、appr_trace）
          vx = torch.where(m & (ex_c < 0) & (ex_c > -tol_), 0.0, vx)
        vy = torch.where(m, torch.where(ey.abs() > dead_, torch.sign(ey) * fvy, 0.0), vy)
        wz = head(torch.full((n,), tyaw, device=dev), m)
        if st == FINE_REJ and a.yaw_first > 0:
          # 向きが a.yaw_first rad 以上ずれている間は、その場で向きだけ直す（前後・横の指令は 0）。取り上げの前に箱が落ちる台の一部は、
          # 寄り始めで 30〜50 度回っていて、回りながら寄って受け具で箱に当たっていた（2026-10-16 夕、appr_trace）
          yw = m & (wrap(tyaw - yaw).abs() > a.yaw_first)
          vx = torch.where(yw, 0.0, vx)
          vy = torch.where(yw, 0.0, vy)
        if st == FINE_REJ and a.retry_wait > 0:
          # 取り上げをやり直す台は、手を構えた姿勢へ開き終わるまで足を止める（開きながら下がると、受け具に掛かった箱を引きずり、
          # やり直した 8 台中 6 台が箱を天板から落とした：2026-10-16、d8）
          rw = m & (n_pick_retry > 0) & (t - t_stage < a.retry_wait)
          vx = torch.where(rw, 0.0, vx)
          vy = torch.where(rw, 0.0, vy)
          wz = torch.where(rw, 0.0, wz)
        ok = m & (ex.abs() <= tol_) & (ey.abs() <= tol_) & (wrap(tyaw - yaw).abs() < math.radians(2.5))
        if st == FINE_REJ and a.retry_wait > 0:
          ok = ok & ~rw  # 手を開き終わるまでは取り上げを始めない
        hold = torch.where(ok | (m & arrived), hold + dt, torch.where(m, 0.0, hold))
        nx = m & (hold > 0.5)
        if a.settle > 0 and st == FINE_REJ:
          # 位置が 0.5 秒合ったら「着いた」とし、指令を 0 にしたまま、止まりきるのを最大 a.settle_max 秒待つ
          # （学習の始めは速度 0 の静止した姿勢）。位置が合い続けることまで求めると、足踏みで許容の外へ出て
          # 寄れない台が 3 割になった（2026-10-13 7:27）
          arrived |= nx
          if not a.settle_correct:
            vx = torch.where(m & arrived, 0.0, vx)
            vy = torch.where(m & arrived, 0.0, vy)
            wz = torch.where(m & arrived, 0.0, wz)
          vlin = torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=-1)
          vang = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=-1)
          still = (vlin < a.settle) & (vang < 0.3)
          if a.settle_qvel > 0:
            # 関節の速さ（全関節のノルム）も小さくなってから取り上げを始める。取り上げで箱を落とした台は、始めた瞬間の関節の速さが
            # 中央値 2.5 rad/s（成功した台は 1.2）で、足踏みの途中から始めていた（2026-10-16）
            still = still & (torch.linalg.norm(robot.data.joint_vel, dim=-1) < a.settle_qvel)
          nx = nx & (still | (hold > 0.5 + a.settle_max))
          if a.settle_correct:
            # 待つ間も位置を直し続け、始める瞬間にも許容の内側であること（指令 0 で待つと前へ流れ、テーブルに 2〜16 cm 近すぎる所から
            # 取り上げを始めて、体が回って箱を落とした：2026-10-13 10:50）
            nx = nx & ok
        if nx.any():
          if st == FINE_REJ:
            # 取り上げの始めの向き（--pick-yaw の成功の条件、取り上げの方策の向きの罰と同じ基準）
            if not hasattr(env, "_fg_yaw0"):
              env._fg_yaw0 = torch.zeros(n, device=dev)
            env._fg_yaw0[nx] = yaw[nx]
            env._fg_table_xy[nx] = torch.tensor([X_REJ, 0.0], device=dev)
            env._fg_table_top[nx] = top_rej
            env._fg_arm_base[nx] = ready_q
          else:
            env._fg_table_xy[nx] = torch.tensor([X_PROD, 0.0], device=dev)
            env._fg_table_top[nx] = top_prod
            env._fg_arm_base[nx] = raise_q
            if getattr(env, "_fg_table_yaw", None) is None:
              env._fg_table_yaw = torch.zeros(n, device=dev)
            env._fg_table_yaw[nx] = math.pi  # 製品テーブルはロボットから見て −x 側（置くの観測の目標の向き）
            env._fg_seated[nx] = False
            env._fg_seat_t[nx] = 0.0
            # 置くに入った瞬間の全状態（製品テーブルの中心・床からの相対。学習の置くの始めの場面に使う：2026-10-13 14:40、
            # 学習の置くの始め（7 割）は箱が受け具に掛かっておらず、0.1 秒で天板へ落ちていた）
            for i in torch.nonzero(nx).flatten().tolist():
              c0 = torch.tensor([X_PROD, 0.0, FLOOR], device=dev)
              entry_place[i] = {"root_pos": (p[i] - c0).tolist(), "root_quat": robot.data.root_link_quat_w[i].tolist(),
                                "root_lin_w": robot.data.root_link_lin_vel_w[i].tolist(), "root_ang_w": robot.data.root_link_ang_vel_w[i].tolist(),
                                "q": robot.data.joint_pos[i].tolist(), "qvel": robot.data.joint_vel[i].tolist(),
                                "box_pos": (s["bp"][i] - c0).tolist(), "box_quat": s["bq"][i].tolist(),
                                "box_lin_w": box.data.root_link_lin_vel_w[i].tolist(), "box_ang_w": box.data.root_link_ang_vel_w[i].tolist(),
                                "top": float(top_prod - FLOOR), "trip": int(trips[i])}
              entry_place_all.append(entry_place[i])  # 往復ごとに上書きされないよう、全部も残す（置くの残差強化学習の始めの場面：2026-10-18）
          at._arm_filt[nx] = at.arm_target[nx]
          if a.filt_actual:
            at._arm_filt[nx] = robot.data.joint_pos[nx][:, env._fg_arm_jids]
          if a.reset_hist:
            # 観測の「前の行動」は action_manager.action を読む（_raw_actions だけでは消えていなかった）
            at._raw_actions[nx] = 0.0
            am_ = env.action_manager
            am_._action[nx] = 0.0
            am_._prev_action[nx] = 0.0
            am_._prev_prev_action[nx] = 0.0
          if st == FINE_REJ:
            for i in torch.nonzero(nx).flatten().tolist():
              entry[i] = {"t": t, "ex": float(ex[i]), "ey": float(ey[i]), "eyaw": float(wrap(tyaw - yaw[i])),
                          "vlin": robot.data.root_link_lin_vel_b[i].tolist(), "vang": robot.data.root_link_ang_vel_b[i].tolist(),
                          "qvel_n": float(torch.linalg.norm(robot.data.joint_vel[i])), "q": robot.data.joint_pos[i].tolist(),
                          "z": float(p[i, 2]), "grav": robot.data.projected_gravity_b[i].tolist(),
                          "arm_err": float((robot.data.joint_pos[i, env._fg_arm_jids] - at.arm_target[i]).abs().max()),
                          "raw_n": float(torch.linalg.norm(at._raw_actions[i])),
                          # 学習の始めの場面に使うための全状態（区画の座標。テーブル・箱は排斥テーブルの決まった位置）
                          "root_pos": p[i].tolist(), "root_quat": robot.data.root_link_quat_w[i].tolist(),
                          "root_lin_w": robot.data.root_link_lin_vel_w[i].tolist(), "root_ang_w": robot.data.root_link_ang_vel_w[i].tolist(),
                          "qvel": robot.data.joint_vel[i].tolist(), "box_pos": s["bp"][i].tolist(), "box_quat": s["bq"][i].tolist()}
            # 因果の確かめ（2026-10-18）：取り上げに入る瞬間に、関節の角速度と胴体の速度だけを変える（姿勢・位置・箱はそのまま）。
            # FG_CF_QVEL=zero：0 にする（完全に落ち着いた始め）、=inject：関節の速さ 3 rad/s 以上だった記録の場面（FG_CF_BANK）から借りる
            cf = os.environ.get("FG_CF_QVEL", "")
            if cf and nx.any():
              ids_ = torch.nonzero(nx).flatten()
              if cf == "zero":
                qv = torch.zeros_like(robot.data.joint_vel[ids_])
                lv = torch.zeros(len(ids_), 6, device=dev)
              else:
                if not hasattr(env, "_fg_cf_bank"):
                  ents_ = torch.load(os.environ["FG_CF_BANK"], weights_only=False)["entry"]
                  env._fg_cf_bank = {k: torch.tensor([e[k] for e in ents_], dtype=torch.float32, device=dev) for k in ("qvel", "root_lin_w", "root_ang_w")}
                bk = env._fg_cf_bank
                jj = torch.randint(0, bk["qvel"].shape[0], (len(ids_),), device=dev)
                qv = bk["qvel"][jj]
                lv = torch.cat([bk["root_lin_w"][jj], bk["root_ang_w"][jj]], dim=-1)
              robot.write_joint_state_to_sim(robot.data.joint_pos[ids_].clone(), qv, env_ids=ids_)
              robot.write_root_link_velocity_to_sim(lv, env_ids=ids_)
          stage[nx] = st + 1
          t_stage[nx] = t
          hold[nx] = 0.0
          arrived[nx] = False
        to = m & (t - t_stage > 30.0)
        stage[to] = FAIL
        for i in torch.nonzero(to).flatten().tolist():
          fail_why[i] = f"{NAMES[st]}：30 秒で寄れない"
          log.append(f"  台{i} {NAMES[st]} 寄れない：前後 {float(ex[i]) * 100:+.1f} cm、横 {float(ey[i]) * 100:+.1f} cm、向き {math.degrees(float(wrap(tyaw - yaw[i]))):+.1f} 度、胴体 x {float(p[i, 0]):+.2f}")
    # 排斥テーブルへ歩く・寄る間の記録（0.2 秒ごと、最後の 40 件：胴体の位置・向き、箱の位置、左右の受け板のずれ）。
    # 取り上げの前に箱が落ちる台の原因を見るため（2026-10-16）
    ap_ = ((stage == WALK_TO_REJ) | (stage == FINE_REJ)) & (k % 10 == 0)
    for i in torch.nonzero(ap_).flatten().tolist():
      e_ = s["hook_err"][i]
      appr_trace[i].append((int(stage[i]), int(trips[i]), round(t - float(t_stage[i]), 1), round(float(p[i, 0]) * 1000), round(float(p[i, 1]) * 1000),
                            round(math.degrees(float(yaw[i]))), round(float(s["bp"][i, 0]) * 1000), round(float(s["bp"][i, 1]) * 1000),
                            round(float(s["bp"][i, 2] - top_rej) * 1000), [round(v * 1000) for v in e_.reshape(-1).tolist()]))
      del appr_trace[i][:-40]
    # 取り上げの前（排斥テーブルへ歩く・寄る）に、箱が排斥テーブルから落ちた台を、その時点で失敗にして記録する。
    # 「取り上げで箱を落とした」12 台のうち 7 台は、取り上げを始めた瞬間に箱が床の上で逆さまだった（2026-10-16）
    pre = ((stage == WALK_TO_REJ) | (stage == FINE_REJ)) & (s["bp"][:, 2] < top_rej - 0.15)
    for i in torch.nonzero(pre).flatten().tolist():
      fail_why[i] = f"{NAMES[int(stage[i])]}：取り上げの前に箱が天板から落ちた"
      log.append(f"  台{i} {NAMES[int(stage[i])]} 取り上げ前の落下：段階に入って {t - float(t_stage[i]):.1f} 秒、往復 {int(trips[i])}、胴体 x {float(p[i, 0]):+.2f} y {float(p[i, 1]):+.2f}、"
                 f"箱 x {float(s['bp'][i, 0]):+.2f} y {float(s['bp'][i, 1]):+.2f}")
    stage[pre] = FAIL
    # --- 取り上げ・置く（V16 の方策）
    m_pick, m_place = stage == PICK, stage == PLACE
    mode = torch.where(m_pick, 1, torch.where(m_place & (not a.place_scripted), 2, mode))
    # 取り上げと、取り上げた後に下がる間の記録（0.1 秒ごと）。取り上げの失敗を「差し込めない／持ち上がらない／持ち上げた後に崩れる／
    # 下がれない」に分けるため（2026-10-16）。受け板のずれ（左右 × 前後・横・上下）、箱の底−天板、傾き、テーブルの中心からの距離、胴体の位置・向き
    m_pt = (m_pick | (stage == BACK_REJ)) & (torch.remainder(torch.round((t - t_stage) / dt), 5) == 0)
    for i in torch.nonzero(m_pt).flatten().tolist():
      e_ = s["hook_err"][i] * 1000
      if pick_trace[i] is None:
        pick_trace[i] = []
      pick_trace[i].append((int(stage[i]), int(trips[i]), round(float(t - t_stage[i]), 2),
                            [round(float(x_)) for x_ in e_[0]], [round(float(x_)) for x_ in e_[1]],
                            round(float(s["gap"][i]) * 1000), round(math.degrees(float(s["tilt"][i]))), round(float(s["dxy"][i]) * 1000),
                            round(float(p[i, 0]) * 1000), round(float(p[i, 1]) * 1000), round(math.degrees(float(yaw[i])), 1)))
    if a.place_v2 and m_place.any():
      tau = t - t_stage
      ez = s["hook_err"][:, :, 2]
      bspd = torch.linalg.norm(s["bv"], dim=-1)
      st0 = m_place & ~ph_started
      if st0.any():
        ph[st0] = 0
        ph_t[st0] = t
        kL[st0] = K0
        kR[st0] = K0
        v2_ok[st0] = 0.0
        set_arm(st0, v2_q("low", kL, kR)[st0], t, 2.0)
        ph_started[st0] = True
        for i in torch.nonzero(st0).flatten().tolist():
          place_trace[i] = [("入る", round(float(s["bp"][i, 0] - X_PROD) * 1000), round(float(s["bp"][i, 1]) * 1000))]
      g = m_place & (ph == 0) & (t - ph_t >= 2.3)
      ph[g] = 1
      ph_t[g] = t - 1.0
      # 1：0.3 秒ごとに測って合わせる
      g = m_place & (ph == 1) & (t - ph_t >= 0.3)
      if g.any():
        seated = (s["gap"].abs() < 0.006) & (bspd < 0.05) & (s["tilt"] < math.radians(6.0))
        # まだ載っていない（箱が受け板の上に乗ったまま）：両方 5 mm 下げる
        down_both = g & ~seated & (kL < 25) & (kR < 25)
        kL[down_both] = torch.clamp(kL[down_both] + 2, max=25)
        kR[down_both] = torch.clamp(kR[down_both] + 2, max=25)
        # 載った：受け板を押さえとの間（フランジの下 0.5〜2.5 mm）へ、片側ずつ 2.5 mm
        hi_l, lo_l = ez[:, 0] > -0.0005, ez[:, 0] < -0.0025
        hi_r, lo_r = ez[:, 1] > -0.0005, ez[:, 1] < -0.0025
        fit = g & seated
        kL[fit & hi_l] = torch.clamp(kL[fit & hi_l] + 1, max=25)
        kL[fit & lo_l] = torch.clamp(kL[fit & lo_l] - 1, min=0)
        kR[fit & hi_r] = torch.clamp(kR[fit & hi_r] + 1, max=25)
        kR[fit & lo_r] = torch.clamp(kR[fit & lo_r] - 1, min=0)
        inband = fit & ~hi_l & ~lo_l & ~hi_r & ~lo_r
        v2_ok = torch.where(inband, v2_ok + 1, torch.where(g, 0.0, v2_ok))
        moved = g & ~inband
        if moved.any():
          set_arm(moved, v2_q("low", kL, kR)[moved], t, 0.2)
        ph_t[g] = t
        for i in torch.nonzero(g).flatten().tolist():
          place_trace[i].append(("測る", round(tau[i].item(), 2), round(ez[i, 0].item() * 1000, 1), round(ez[i, 1].item() * 1000, 1),
                                 round(s["gap"][i].item() * 1000), round(bspd[i].item(), 3), int(kL[i]), int(kR[i]),
                                 # 手首の高さ（区画の座標、mm）と、腕の目標と実際の関節の差の最大（rad）
                                 [round(float(v) * 1000) for v in (robot.data.body_link_pos_w[i, wrist_ids, 2] - org[i, 2])],
                                 round(float((robot.data.joint_pos[i, env._fg_arm_jids] - at.arm_target[i]).abs().max()), 3),
                                 # 胴体の高さ・前後の傾き、左肩のピッチの目標と実際
                                 round(float(p[i, 2]) * 1000), round(math.degrees(float(torch.asin(torch.clamp(robot.data.projected_gravity_b[i, 0], -1, 1)))), 1),
                                 round(float(at.arm_target[i, 0]), 3), round(float(robot.data.joint_pos[i, env._fg_arm_jids[0]]), 3),
                                 # 上体（waist_pitch_link）の高さ、左肘の目標と実際
                                 round(float(robot.data.body_link_pos_w[i, torso_id, 2] - org[i, 2]) * 1000),
                                 round(float(at.arm_target[i, elbow_l]), 3), round(float(robot.data.joint_pos[i, env._fg_arm_jids[elbow_l]]), 3)))
        # 2 回続けて範囲内 → 開く（同じ高さのまま外へ、1.2 秒）。合わせるのは 6 秒まで（時間切れでも開かず、15 秒で失敗にする）
        go = g & (v2_ok >= 2)
        if go.any():
          set_arm(go, v2_q("open", kL, kR)[go], t, 1.2)
          ph[go] = 2
          ph_t[go] = t
          for i in torch.nonzero(go).flatten().tolist():
            place_trace[i].append(("開く", round(tau[i].item(), 2), int(kL[i]), int(kR[i])))
      # 2：両手が外へ抜けたら上げる。抜けないうちは上げない（引っかけたまま上げると箱を持ち上げて落とす）
      g = m_place & (ph == 2) & (t - ph_t >= 1.3) & pl._released(s)
      if g.any():
        set_arm(g, v2_q("up", kL, kR)[g], t, 0.6)
        ph[g] = 3
        ph_t[g] = t
      g = m_place & (ph == 3) & (t - ph_t >= 0.6)
      ph[g] = 4
    elif a.place_closed and m_place.any():
      tau = t - t_stage
      ez = s["hook_err"][:, :, 2]
      bspd = torch.linalg.norm(s["bv"], dim=-1)
      # 入った最初：抱える姿勢（3 cm 上げた）から下ろす（2 秒）
      st0 = m_place & ~ph_started
      if st0.any():
        ph[st0] = 0
        ph_t[st0] = t
        n_retry[st0] = 0
        sel_l[st0] = 0
        sel_r[st0] = 0
        rel_on[st0] = False
        # 因果の確かめ（2026-10-18）：置く間に腰が横へ流れる元を分ける。FG_CF_PLACE=noarm：腕を下ろさない（抱える姿勢のまま）、
        # =nobox：置くに入った瞬間に箱を区画の外へ移し、腕だけ下ろす。既定は今まで通り
        cfp = os.environ.get("FG_CF_PLACE", "")
        # =stand：箱を外し、腕も下ろさない（歩行の方策が指令 0 で立っているだけ）
        if cfp not in ("noarm", "stand"):
          set_arm(st0, low_q, t, a.place_low_dur)
        if cfp in ("nobox", "stand"):
          ids_ = torch.nonzero(st0).flatten()
          far_ = org[ids_] + torch.tensor([X_PROD, 3.0, 0.3], device=dev)
          box.write_root_link_pose_to_sim(torch.cat([far_, torch.tensor([1.0, 0, 0, 0], device=dev).expand(len(ids_), 4)], dim=-1), env_ids=ids_)
          box.write_root_link_velocity_to_sim(torch.zeros(len(ids_), 6, device=dev), env_ids=ids_)
        ph_started[st0] = True
        for i in torch.nonzero(st0).flatten().tolist():
          place_trace[i] = [("入る", round(float(s["bp"][i, 0] - X_PROD) * 1000), round(float(s["bp"][i, 1]) * 1000))]
        hold_xy[st0] = p[st0, :2]
        open_xy[st0] = p[st0, :2]
        hold_yaw[st0] = yaw[st0]
      if a.place_hold > 0:
        # 腕を下ろす間、胴体が +y へ横に流れ（外れる台は 0.3〜1.2 秒で 18 mm）、箱がテーブルの中心から外れていた。
        # 置く間は歩行の指令が 0 で、流れを戻す仕組みが無かった（2026-10-15）。
        # 箱が天板に着いた後は足を動かさない（踏み替えで、載った箱を引きずるため）
        hm = m_place & ((ph == 0) | ((ph == 1) & a.place_hold_ph1)) & (s["gap"] > 0.008)
        eh = hold_xy - p[:, :2]
        if a.place_hold_box:
          # 胴体ではなく箱をテーブルの中心へ寄せる。胴体を保っても、外れる台は箱が胴体に対して +2 cm 動いていた（2026-10-15 夜）
          eh = torch.tensor([X_PROD, 0.0], device=dev) - s["bp"][:, :2]
        hx = c_ * eh[:, 0] + s_ * eh[:, 1]
        hy = -s_ * eh[:, 0] + c_ * eh[:, 1]
        # 速さは寄る段階と同じ（これより遅いと方策が足を出さない：2026-10-13 3:20）
        vx = torch.where(hm & (hx.abs() > a.place_hold), torch.where(hx > 0, 0.12, -0.2), vx)
        vy = torch.where(hm & (hy.abs() > a.place_hold), torch.sign(hy) * 0.2, vy)
        wz = head(hold_yaw, hm)
      # 下ろして合わせる間の記録（0.1 秒ごと：箱・胴体の位置、受け板の左右の外へのずれ、箱の底−天板）。
      # 外れる台は、この間に箱が +y へ約 5 cm 流れていた。胴体が横へ流れるのか、片手が押すのかを分ける（2026-10-15）
      og0 = m_place & (ph <= 1) & (torch.remainder(torch.round(tau / dt), 5) == 0)
      for i in torch.nonzero(og0).flatten().tolist():
        e_ = s["hook_err"][i]
        place_trace[i].append(("合わせる途中", round(float(tau[i]), 2), int(ph[i]),
                               round(float(s["bp"][i, 0] - X_PROD) * 1000), round(float(s["bp"][i, 1]) * 1000),
                               round(float(p[i, 0] - X_PROD) * 1000), round(float(p[i, 1]) * 1000),
                               round(float(e_[0, 1]) * 1000), round(float(e_[1, 1]) * 1000), round(s["gap"][i].item() * 1000)))
      if a.open_hold and a.place_hold > 0:
        # 開いてから終わるまで、開き始めた時の胴体の位置・向きを保つ。左手が抜けない台は、開いて上げた後に胴体が +y へ約 3 cm 流れ、
        # 外へ抜けた左手が箱の上へ戻っていた（成功した台は −y へ流れていた：2026-10-16、int_r28_30500_3trips.pt の place_trace）
        ho = m_place & (ph >= 2)
        eo = open_xy - p[:, :2]
        ox = c_ * eo[:, 0] + s_ * eo[:, 1]
        oy = -s_ * eo[:, 0] + c_ * eo[:, 1]
        vx = torch.where(ho & (ox.abs() > a.place_hold), torch.where(ox > 0, 0.12, -0.2), vx)
        vy = torch.where(ho & (oy.abs() > a.place_hold), torch.sign(oy) * 0.2, vy)
        wz = torch.where(ho, head(open_yaw, ho), wz)
      # 上げた後の記録（0.2 秒ごと、上げ始めから 3 秒まで：受け板の左右の外へのずれ、胴体の位置）
      og3 = m_place & (ph >= 3) & (t - ph_t < 3.0) & (torch.remainder(torch.round((t - ph_t) / dt), 10) == 0)
      for i in torch.nonzero(og3).flatten().tolist():
        e_ = s["hook_err"][i]
        place_trace[i].append(("上げた後", int(ph[i]), round(float(t - ph_t[i]), 2), round(float(e_[0, 1]) * 1000), round(float(e_[1, 1]) * 1000),
                               round(float(p[i, 0] - X_PROD) * 1000), round(float(p[i, 1]) * 1000)))
      # 0 → 1：下ろし終えて 0.4 秒待った
      g = m_place & (ph == 0) & (t - ph_t >= a.place_low_dur + 0.4)
      ph[g] = 1
      ph_t[g] = t - 1.0  # すぐ 1 回目を測る
      # 1：0.4 秒ごとに測る。受け板がフランジの下 1.2 cm より上（重さが掛かりうる）側だけ 1 cm 下げる（最大 4 cm）
      g = m_place & (ph == 1) & (t - ph_t >= 0.4)
      if g.any():
        # 受け板 0・1 と逆運動学の左・右の対応（--place-swap で入れ替えて確かめる：2026-10-13 7:46、右を下げると受け板 0 が下がる記録があった）
        iL, iR = (1, 0) if a.place_swap else (0, 1)
        # 追加で下げるのは a.place_max_sel cm まで（下げるほど手が前後に傾き、両方掛かった箱では受け具が外れなくなった：2026-10-13 14:10）
        need_l = (ez[:, iL] > -0.012) & (sel_l < a.place_max_sel)
        need_r = (ez[:, iR] > -0.012) & (sel_r < a.place_max_sel)
        seated = (s["gap"].abs() < 0.008) & (bspd < 0.05)
        adj = g & (need_l | need_r)
        dl_ = rel_delta(s["bp"], s["bq"])
        for i in torch.nonzero(g).flatten().tolist():
          place_trace[i].append(("測る", round(tau[i].item(), 2), round(ez[i, 0].item() * 1000), round(ez[i, 1].item() * 1000),
                                 round(s["gap"][i].item() * 1000), round(bspd[i].item(), 3),
                                 # 箱の上体に対するずれ（位置 mm、回転 度：上体の x・y・z 軸まわり）。2026-10-17 追加
                                 *[round(float(v_) * 1000) for v_ in dl_[i, :3]], *[round(math.degrees(float(v_)), 1) for v_ in dl_[i, 3:]]))
        if a.place_rel > 0:
          # 箱が天板の上で止まっていれば、その位置・向きを基準にする（手は箱に触れているので、箱そのものを毎回の基準にすると、手が箱を押し上げて目標も上がる）
          st_ = g & (s["gap"].abs() < 0.008) & (bspd < 0.05)
          rel_bp[st_] = s["bp"][st_]
          rel_bq[st_] = s["bq"][st_]
          rel_on[st_] = True
        sel_l[adj & need_l] += 1
        sel_r[adj & need_r] += 1
        if adj.any():
          set_arm(adj, low2_tab[sel_l, sel_r][adj], t, 0.3)
          ph_t[adj] = t
        # 左右とも抜けた（または 4 cm まで下げた）、かつ箱が天板に載って止まっている → 開く。合わせるのは 4 秒まで
        go = (g & ~adj & seated) | (m_place & (ph == 1) & (tau > a.place_low_dur + 0.4 + a.place_adj_max))
        ph_t[g & ~adj & ~go] = t  # 次に測るのは 0.4 秒後
        if go.any():
          set_arm(go, open2_tab[sel_l, sel_r][go], t, a.open_dur)
          ph[go] = 2
          ph_t[go] = t
          open_xy[go] = p[go, :2]
          open_yaw[go] = yaw[go]
          for i in torch.nonzero(go).flatten().tolist():
            place_trace[i].append(("開く", round(tau[i].item(), 2), int(sel_l[i]), int(sel_r[i])))
      if res_on and not a.residual_no_open:
        # 置くの残差の方策の「もう開いてよい」の合図（載った判定・4 秒の時間切れを待たずに開く）
        og_ = m_place & (ph == 1) & res_open
        if og_.any():
          set_arm(og_, open2_tab[sel_l, sel_r][og_], t, a.open_dur)
          ph[og_] = 2
          ph_t[og_] = t
          open_xy[og_] = p[og_, :2]
          open_yaw[og_] = yaw[og_]
          for i in torch.nonzero(og_).flatten().tolist():
            if place_trace[i] is not None:
              place_trace[i].append(("開く（合図）", round(tau[i].item(), 2), int(sel_l[i]), int(sel_r[i])))
      # 2：開き終えて、両手が外へ抜けた → 上げる。抜けないまま 2.5 秒で上げる
      # 開いている間の記録（0.1 秒ごと：受け板の左右の外へのずれ・上下、箱の位置・傾き）
      og = m_place & (ph == 2) & (torch.remainder(torch.round((t - ph_t) / dt), 5) == 0)
      for i in torch.nonzero(og).flatten().tolist():
        e_ = s["hook_err"][i]
        place_trace[i].append(("開く途中", round(float(t - ph_t[i]), 2), round(float(e_[0, 1]) * 1000), round(float(e_[1, 1]) * 1000),
                               round(float(e_[0, 2]) * 1000), round(float(e_[1, 2]) * 1000),
                               round(float(s["bp"][i, 0] - X_PROD) * 1000), round(float(s["bp"][i, 1]) * 1000), round(math.degrees(float(s["tilt"][i]))),
                               # 腕の目標と実際の関節の差（左の腕・右の腕の最大、rad）
                               round(float((robot.data.joint_pos[i, env._fg_arm_jids] - at._processed_actions[i, at._arm_idx])[is_l_arm].abs().max()), 3),
                               round(float((robot.data.joint_pos[i, env._fg_arm_jids] - at._processed_actions[i, at._arm_idx])[~is_l_arm].abs().max()), 3),
                               # 胴体の位置（mm、2026-10-15 追加）
                               round(float(p[i, 0] - X_PROD) * 1000), round(float(p[i, 1]) * 1000)))
      if a.place_retry > 0:
        # 開いて 1.3 秒たっても抜けない手は、その手だけ 1 cm 下げて開き直す（最大 a.place_retry 回）。
        # 片手（主に左）が押さえとフランジに挟まって抜けない台が 1 往復あたり 6〜9 台残っていた（2026-10-13 19:00）
        e_ = s["hook_err"]
        out_l, out_r = e_[:, 0, 1] > 0.05, e_[:, 1, 1] < -0.05
        rt = m_place & (ph == 2) & (t - ph_t >= 1.3) & ~(out_l & out_r) & (n_retry < a.place_retry)
        if rt.any():
          sel_l[rt & ~out_l] = torch.clamp(sel_l[rt & ~out_l] + 1, max=4)
          sel_r[rt & ~out_r] = torch.clamp(sel_r[rt & ~out_r] + 1, max=4)
          # 抜けた手は開いた姿勢のまま、抜けない手だけ下げた構えへ（両手を閉じ直すと、抜けた手がまた掛かった）
          q_lo, q_op = low2_tab[sel_l, sel_r], open2_tab[sel_l, sel_r]
          keep_l = out_l[:, None] & is_l_arm[None, :]
          keep_r = out_r[:, None] & ~is_l_arm[None, :]
          q_rt = torch.where(keep_l | keep_r, q_op, q_lo)
          set_arm(rt, q_rt[rt], t, 0.3)
          n_retry[rt] += 1
          ph[rt] = 1
          ph_t[rt] = t - 0.1  # 0.3 秒後に測って開き直す
          for i in torch.nonzero(rt).flatten().tolist():
            place_trace[i].append(("開き直す", round(float(t - t_stage[i]), 2), int(sel_l[i]), int(sel_r[i])))
      g = m_place & (ph == 2) & (t - ph_t >= a.open_dur + 0.1) & (pl._released(s) | (t - ph_t >= 2.5 + 10.0 * (a.place_retry > 0)))
      if g.any():
        set_arm(g, up2_tab[sel_l, sel_r][g], t, 0.6)
        ph[g] = 3
        ph_t[g] = t
        # 時間の内訳の記録（2026-10-17）：上げ始め、両手が抜けたか
        rel_ = pl._released(s)
        for i in torch.nonzero(g).flatten().tolist():
          place_trace[i].append(("上げる", round(float(tau[i]), 2), bool(rel_[i])))
      g = m_place & (ph == 3) & (t - ph_t >= 0.6)
      ph[g] = 4
      for i in torch.nonzero(g).flatten().tolist():
        place_trace[i].append(("段階4", round(float(tau[i]), 2)))
    if a.place_closed or a.place_v2:
      ph_started[~m_place] = False
    elif a.place_scripted and m_place.any():
      # 下ろす（2 秒）→ 待つ（0.5 秒）→ 外へ開く（1 秒）→ 2 cm 上げる（0.6 秒）
      tau = t - t_stage
      # 2.4 秒で左右の受け板の高さを測る（受け板の中心のずれの上下。フランジの下 1.2 cm まで下げたい）
      dec = m_place & (tau >= 2.4) & ~decided
      if dec.any():
        ez = s["hook_err"][:, :, 2]
        need = torch.clamp(ez + 0.012, 0.0, 0.04)
        sel_l[dec] = torch.ceil(need[dec, 0] / 0.01 - 1e-6).long().clamp(0, 4)
        sel_r[dec] = torch.ceil(need[dec, 1] / 0.01 - 1e-6).long().clamp(0, 4)
        decided[dec] = True
        for i in torch.nonzero(dec).flatten().tolist()[:6]:
          log.append(f"  台{i} 置く：受け板の上下のずれ 左 {float(ez[i, 0]) * 1000:+.0f} 右 {float(ez[i, 1]) * 1000:+.0f} mm → 追加で下げる 左 {int(sel_l[i])} 右 {int(sel_r[i])} cm")
      decided[~m_place] = False
      low2 = low2_tab[sel_l, sel_r]
      open2 = open2_tab[sel_l, sel_r]
      up2 = up2_tab[sel_l, sel_r]
      for t0, t1, q0, q1 in ((0.0, 2.0, raise_q, low_q), (2.4, 2.9, low_q, low2), (2.9, 3.9, low2, open2), (3.9, 4.5, open2, up2)):
        mm_ = m_place & (tau >= t0)
        if mm_.any():
          u = torch.clamp((tau - t0) / (t1 - t0), 0.0, 1.0)
          u = u * u * (3 - 2 * u)
          q = q0 + (q1 - q0) * u[:, None] if q0.dim() == q1.dim() else (q0 + (q1 - q0) * u[:, None])
          arm_from[mm_] = q[mm_]
          arm_to[mm_] = q[mm_]
    # 受け板のずれの許容（既定 2 cm）。取り上げの 15 秒切れの多くは、箱を 4 cm 持ち上げて傾き 3 度なのに、
    # 受け板のずれの平均が 2.0〜2.7 cm で判定に届かなかった（成功した台も 1.5〜1.8 cm）：2026-10-16
    pick_ok = m_pick & (s["hook_dist"] < a.pick_hook_tol) & (s["gap"] > 0.02) & (s["tilt"] < math.radians(10.0)) & s["standing"]
    if a.pick_dxy > 0:
      # 箱がテーブルの上（中心から a.pick_dxy m 以内）で持ち上がったことも条件に。取り上げの方策が横へ 30〜50 cm 歩いて、
      # 箱を天板の端から引きずり出して「浮いた」ことにし、下がり始めて落とす台があった（2026-10-13 19:40）
      pick_ok = pick_ok & (s["dxy"] < a.pick_dxy)
    if a.pick_yaw:
      # 向きを保って持ち上げたことも条件に（体ごと 30〜60 度回ったまま成功にして、下がる間に落としていた）
      pick_ok = pick_ok & (pl._pick_yaw_err(env, s).abs() < math.radians(pl.PICK_YAW_TOL)) & (robot.data.root_link_ang_vel_w[:, 2].abs() < pl.PICK_WZ_TOL)
    if m_place.any():
      pl._update_seated(env, s)
    if a.place_closed or a.place_v2:
      place_ok = m_place & (ph >= 4) & pl._resting(s) & pl._released(s) & s["standing"]
    elif a.place_scripted:
      place_ok = m_place & (t - t_stage > 4.6) & pl._resting(s) & pl._released(s) & s["standing"]
    else:
      place_ok = m_place & pl._resting(s) & pl._released(s) & env._fg_seated & s["standing"]
    okm = pick_ok | place_ok
    hold = torch.where(okm, hold + dt, torch.where(m_pick | m_place, 0.0, hold))
    nx = (m_pick & (hold > 1.0)) | (m_place & (hold > 0.5))
    if (a.place_closed or a.place_v2) and (nx & m_place).any():
      for i in torch.nonzero(nx & m_place).flatten().tolist():
        if place_trace[i] is not None:
          place_trace[i].append(("置き終わり", round(float(t - t_stage[i]), 2)))
    if nx.any():
      at.arm_target[nx] = at._processed_actions[nx][:, at._arm_idx]
      back_from[nx] = p[nx, :2]
      back_yaw[nx] = yaw[nx]
      # 抱える姿勢へ移るのを a.carry_delay 秒遅らせる（持ち上げた姿勢のまま下がり始め、箱がテーブルから離れてから下ろす）。
      # 取り上げ後に下がれない台は、抱える姿勢へ下ろした箱がテーブルの縁に載ったまま 20 cm で止まっていた（2026-10-16）
      if a.carry_after_back > 0:
        # 抱える姿勢へは、a.carry_after_back m 下がって箱がテーブルから離れてから移る（それまでは取り上げの最後の目標を保つ（2026-10-16）
        mk_ = nx & m_pick
        set_arm(mk_, at.arm_target[mk_].clone(), t, 0.05)
        carry_pending[mk_] = True
      else:
        set_arm(nx & m_pick, carry_q, t + a.carry_delay, a.carry_dur)
      # 置いた後は、開いた手をそのまま（シーケンス制御なら 2 cm 上げた開いた構え）で下がる。構えた姿勢（1 cm 下）へ戻しながら下がると、
      # 手が箱・天板に引っかかって下がれなかった（2026-10-13 3:20）。構えた姿勢へは振り返るときに戻す
      if a.place_scripted:
        mp_ = nx & m_place
        arm_from[mp_] = at.arm_target[mp_] - res_add[mp_]
        arm_to[mp_] = (v2_q("up", kL, kR) if a.place_v2 else up2_tab[sel_l, sel_r])[mp_]
        arm_t0[mp_] = t
        arm_dur[mp_] = 0.3
      stage[nx] = stage[nx] + 1
      t_stage[nx] = t
      hold[nx] = 0.0
    if a.pick_retry > 0:
      # 取り上げを始めて a.pick_retry 秒たっても箱が 1.5 cm 浮いていない台は、寄る段階へ戻ってやり直す（1 往復につき 1 回）。
      # 取り上げの 15 秒切れの 6 割は、受け板が 3 cm ずれたまま箱を 1〜9 mm しか持ち上げられず、そのまま 15 秒が過ぎていた（2026-10-16、d4 の pick_trace）
      rr = m_pick & (t - t_stage > a.pick_retry) & (s["gap"] < 0.015) & (n_pick_retry < 1) & s["standing"]
      if rr.any():
        at.arm_target[rr] = at._processed_actions[rr][:, at._arm_idx]
        set_arm(rr, ready_q, t, 1.0)
        n_pick_retry[rr] += 1
        stage[rr] = FINE_REJ
        t_stage[rr] = t
        hold[rr] = 0.0
        arrived[rr] = False
        for i in torch.nonzero(rr).flatten().tolist():
          log.append(f"  台{i} 取り上げ やり直し：箱の底−天板 {float(s['gap'][i]) * 1000:.0f} mm、受け板のずれ {float(s['hook_dist'][i]) * 1000:.0f} mm")
        m_pick = m_pick & ~rr
    to = (m_pick | m_place) & (t - t_stage > 15.0)
    for i in torch.nonzero(to).flatten().tolist():
      why_ = []
      if int(stage[i]) == PLACE:
        if not bool(s["gap"][i].abs() < 0.006):
          why_.append("載っていない")
        if not bool(s["tilt"][i] < math.radians(6.0)):
          why_.append("傾き")
        if not bool(s["dxy"][i] < 0.08):
          why_.append("中心から外れた")
        e_ = s["hook_err"][i]
        if not bool(e_[0, 1] > 0.05):
          why_.append("左手が抜けない")
        if not bool(e_[1, 1] < -0.05):
          why_.append("右手が抜けない")
        if not bool(s["standing"][i]):
          why_.append("立っていない")
      fail_why[i] = f"{NAMES[int(stage[i])]}：15 秒で終わらない" + (f"（{'・'.join(why_)}）" if why_ else "")
      log.append(f"  台{i} {NAMES[int(stage[i])]} 終わらず：箱の底−天板 {float(s['gap'][i]) * 1000:.0f} mm、傾き {math.degrees(float(s['tilt'][i])):.0f} 度、"
                 f"テーブルの中心から {float(s['dxy'][i]) * 1000:.0f} mm、受け板のずれ {[round(v, 3) for v in s['hook_err'][i].reshape(-1).tolist()]}")
    stage[to] = FAIL
    # --- 下がる（0.4 m）
    for st in (BACK_REJ, BACK_PROD):
      m = stage == st
      if m.any():
        # −0.3 m/s（run14 の学習の範囲の下限）。−0.2 では 10 秒で 9〜19 cm しか下がれなかった（2026-10-13 3:25）
        # 取り上げた後は、腕が抱える姿勢になりきってから下がる（--back-wait、既定 0.5 秒。腕を動かしながら下がって箱を落としていた）
        bw = a.back_wait if st == BACK_REJ else 0.5
        if st == BACK_REJ and a.back_stable:
          # 腕が抱える姿勢になり、箱が掛かって傾かず、体の回りが止まってから下がる（最長 3 秒待つ）
          st_ok = (t - t_stage > a.carry_dur) & (s["hook_dist"] < 0.03) & (s["tilt"] < math.radians(6.0)) & (robot.data.root_link_ang_vel_w[:, 2].abs() < 0.2)
          go_b = st_ok | (t - t_stage > 3.0)
          vx = torch.where(m & go_b, -0.3, vx)
          # 下がる間も、取り上げの始めの向きを保つ（取り上げ中に体が回った台を、テーブルから離れながら戻す）
          if hasattr(env, "_fg_yaw0"):
            wz = head(env._fg_yaw0, m & go_b)
        else:
          vx = torch.where(m & (t - t_stage > bw), -0.3, vx)
          if st == BACK_PROD and a.back_hold:
            # 置いた後に下がる間も、下がり始めた時の横の位置と向きを保つ（--open-hold は置くが終わると切れ、下がる間は前後の指令だけだった。
            # 置いた箱が下がる間に落ちる台が 4〜9 台残る（2026-10-16 午後）
            eb = back_from - p[:, :2]
            by_ = -s_ * eb[:, 0] + c_ * eb[:, 1]
            vy = torch.where(m & (by_.abs() > 0.02), torch.sign(by_) * 0.2, vy)
            wz = torch.where(m, head(back_yaw, m), wz)
        # 置いた後は a.back_prod m 下がる（0.3 m では、開いた手で振り返るときに置いた箱を天板から払い落とした：2026-10-13 18:00）
        back_d = a.back_prod if st == BACK_PROD else 0.3
        if st == BACK_REJ and a.carry_after_back > 0:
          cg = m & carry_pending & (torch.linalg.norm(p[:, :2] - back_from, dim=-1) > a.carry_after_back)
          if cg.any():
            set_arm(cg, carry_q, t, a.carry_dur)
            carry_pending[cg] = False
        nx = m & (torch.linalg.norm(p[:, :2] - back_from, dim=-1) > back_d)
        stage[nx] = st + 1
        t_stage[nx] = t
        to = m & (t - t_stage > 20.0)
        for i in torch.nonzero(to).flatten().tolist():
          fail_why[i] = f"{NAMES[st]}：下がれない"
          log.append(f"  台{i} {NAMES[st]} 下がれない：下がった距離 {float(torch.linalg.norm(p[i, :2] - back_from[i])) * 100:.1f} cm、受け板のずれ {[round(v, 3) for v in s['hook_err'][i].reshape(-1).tolist()]}")
        stage[to] = FAIL
    # --- 振り返る（前へ歩きながら回る）
    for st, tyaw in ((TURN_REJ, math.pi), (TURN_PROD, 0.0)):
      m = stage == st
      if m.any():
        if st == TURN_PROD:
          # 置いた後の振り返りは、その場で回る（前へ歩きながら回ると、開いた手で置いた箱を天板から払い落とした：256 台中 45 台）。
          # 腕は回り終わってから構えた姿勢へ戻す
          pass
        e = wrap(tyaw - yaw)
        if st == TURN_REJ and a.turn_rej:
          # 台の上（箱を持つ）は、製品テーブルの前と別の上限・下限（速く回ると箱を落とした：2026-10-17）
          w_ = torch.sign(e) * torch.clamp(1.5 * e.abs(), a.turn_rej_wmin, a.turn_rej_wmax)
        else:
          w_ = torch.sign(e) * torch.clamp(1.5 * e.abs(), a.turn_wmin, a.turn_wmax if st == TURN_PROD else a.turn_rej_loco_wmax)
        if st == TURN_PROD and a.turn_ramp > 0:
          # 置いた後の振り返りは、下がり終えてから回る指令を a.turn_ramp 秒かけて上げる（下がる −0.3 m/s からいきなり 0.5 rad/s で回り始め、
          # 製品テーブルの前で転ぶ台が 3 往復で 5 台あった（2026-10-16）
          w_ = w_ * torch.clamp((t - t_stage) / a.turn_ramp, 0.0, 1.0)
        wz = torch.where(m, w_, wz)
        vx = torch.where(m & (e.abs() > 0.15), 0.12 if (st == TURN_REJ and not a.turn_rej) else 0.0, vx)  # 回れ右の方策（--turn-rej）は、その場で回る（前へ歩く指令は学習していない）
        if os.environ.get("FG_DBG_TURN") and k % 10 == 0:
          # 振り返りの調べ：最初の 4 台の向きの残り・指令・実際の回る速さ（2026-10-17）
          for i in torch.nonzero(m).flatten().tolist()[:4]:
            log.append(f"  回る {NAMES[st][:4]} 台{i} {t - float(t_stage[i]):5.1f} 秒 残り {math.degrees(float(e[i])):+6.1f} 度 指令 {float(wz[i]):+.2f} 実際 {float(robot.data.root_link_ang_vel_b[i, 2]):+.2f} rad/s")
        nx = m & (e.abs() < 0.06)
        if st == TURN_PROD and nx.any():
          # 完了・次の往復へ移る瞬間にも、置いた箱が天板の上にあることを確かめる（この 1 歩が抜けていた）
          on_tab0 = (s["gap"].abs() < 0.015) & (torch.linalg.norm(s["bp"][:, :2] - torch.tensor([X_PROD, 0.0], device=dev), dim=-1) < 0.14) & (s["tilt"] < math.radians(10.0))
          lost0 = nx & ~on_tab0
          for i in torch.nonzero(lost0).flatten().tolist():
            fail_why[i] = f"{NAMES[st]}：置いた箱が天板から落ちた"
          stage[lost0] = FAIL
          nx = nx & ~lost0
          trips[nx] += 1
          done = nx & (trips >= a.trips)
          t_done[done] = t
          stage[done] = DONE
          nx = nx & ~done
          spawn_box(nx)  # 次の往復：新しいケースを排斥テーブルへ
          n_pick_retry[nx] = 0
          set_arm(nx, ready_q, t, 1.0)
        stage[nx] = (st + 1) % 10
        t_stage[nx] = t
    # 置いた後（下がる・振り返る）は、箱が製品テーブルの天板の上に残っていること（手を離した後に引っかけて落とした台を
    # 成功に数えていた：3 往復の動画の最後で箱が床にあった、2026-10-13 4:40）
    after = (stage == BACK_PROD) | (stage == TURN_PROD)
    on_tab = (s["gap"].abs() < 0.015) & (torch.linalg.norm(s["bp"][:, :2] - torch.tensor([X_PROD, 0.0], device=dev), dim=-1) < 0.14) & (s["tilt"] < math.radians(10.0))
    lost = after & ~on_tab
    for i in torch.nonzero(lost).flatten().tolist():
      fail_why[i] = f"{NAMES[int(stage[i])]}：置いた箱が天板から落ちた"
    stage[lost] = FAIL
    # 箱を落とした（運んでいる間）：箱が床・台より大きく下、または 45 度以上傾いた
    carrying = (stage >= BACK_REJ) & (stage <= PLACE)
    dropped = carrying & ((s["tilt"] > math.radians(45.0)) | (s["bp"][:, 2] < p[:, 2] - 0.2))
    if os.environ.get("FG_CF_PLACE", "") in ("nobox", "stand"):
      dropped = dropped & (stage != PLACE)  # 因果の確かめで箱を外した「置く」の間は、落下と判定しない（腰の流れを測るため）
    # 取り上げの間も、学習・単体の評価と同じ落下の判定（天板より 10 cm 下・45 度・12 cm 超投げ上げ）。これが無いと、
    # 箱を落とした後も取り上げの方策を動かし続けて転び、「取り上げで転倒」に数えていた（2026-10-13 8:32）
    # （入った直後の 1 歩は、天板の高さ env._fg_table_top をこの歩で切り替えたので s が古い。2 往復目に、製品テーブルの高さで測って
    # 「39 cm 投げ上げた」と誤判定していた：2026-10-13 10:30）
    dropped = dropped | ((stage == PICK) & (t - t_stage > 0.1) & ((s["tilt"] > math.radians(45.0)) | (s["gap"] < -0.1) | (s["gap"] > 0.12)))
    for i in torch.nonzero(dropped & (stage != FAIL)).flatten().tolist():
      fail_why[i] = f"{NAMES[int(stage[i])]}：箱を落とした"
      log.append(f"  台{i} {NAMES[int(stage[i])]} 落下：往復 {int(trips[i])}、段階に入って {t - float(t_stage[i]):.1f} 秒、箱の底−天板 {float(s['gap'][i]) * 1000:.0f} mm、"
                 f"傾き {math.degrees(float(s['tilt'][i])):.0f} 度、箱 x {float(s['bp'][i, 0]):+.2f} y {float(s['bp'][i, 1]):+.2f} z {float(s['bp'][i, 2]):+.2f}、天板 {float(env._fg_table_top[i]):+.3f}、"
                 f"胴体 x {float(p[i, 0]):+.2f} y {float(p[i, 1]):+.2f} 向き {math.degrees(float(yaw[i])):+.0f} 度")
      if (a.place_closed or a.place_v2) and int(stage[i]) == PLACE and place_trace[i] is not None:
        place_trace[i].append(("落とした", round(t - float(t_stage[i]), 2), int(ph[i])))
    stage[dropped] = FAIL
    # 腕の目標（歩く方策の間）
    aa = torch.clamp((t - arm_t0) / arm_dur, 0.0, 1.0)
    aa = aa * aa * (3 - 2 * aa)
    manip_m = (mode > 0)
    at.arm_target = torch.where(manip_m[:, None], at.arm_target, arm_from + (arm_to - arm_from) * aa[:, None])
    if a.place_scripted and a.place_rel > 0:
      # 置くの間、測った箱を基準に、今の上体の姿勢から腕の目標を直す（滑らかに：時定数 約 0.2 秒）
      rm = rel_on & (stage == PLACE) & (ph >= 1) & (ph <= 3)
      want = torch.zeros_like(rel_add)
      if rm.any():
        dl_ = rel_delta(rel_bp, rel_bq)
        Jm = torch.where((ph >= 2)[:, None, None], J_open, J_low)
        want = a.place_rel * torch.einsum("nkj,nj->nk", Jm, dl_) * rm[:, None]
      rel_add = torch.where(rm[:, None], rel_add + 0.1 * (want - rel_add), torch.zeros_like(rel_add))
      at.arm_target = at.arm_target + rel_add
    if res_on:
      at.arm_target = at.arm_target + res_add
    env._fg_mode[:] = mode
    if a.lock_waist_place:
      # 置くの間（シーケンス制御）は腰の 3 軸（または pitch だけ）を 0 に固定
      at.waist_target = torch.zeros(n, 3, device=dev)
      mp_ = (stage == PLACE)
      at.waist_mask = torch.stack([mp_ & (a.lock_waist_place == "all")] * 2 + [mp_], dim=-1)
    stopped = (stage == DONE) | (stage == FAIL)
    cmd.vel_command_b[:, 0] = torch.where(stopped | manip_m, 0.0, vx)
    cmd.vel_command_b[:, 1] = torch.where(stopped | manip_m, 0.0, vy)
    cmd.vel_command_b[:, 2] = torch.where(stopped | manip_m, 0.0, wz)
    if a.fresh_obs:
      # mjlab は観測をキャッシュするので、捨ててから作り直す（履歴・遅延のバッファはこの課題では使っていない）
      env.observation_manager._obs_buffer = None
      obs = wenv.get_observations()
    else:
      obs = wenv.get_observations() if k == 0 else obs
    with torch.no_grad():
      act_l = loco(obs["actor"])
      if climb_pol is not None:
        cm_ = (stage == WALK_TO_REJ) | ((stage == WALK_TO_PROD) & a.climb_desc)
        if cm_.any():
          act_l = torch.where(cm_[:, None], climb_pol(obs["actor"]), act_l)
      if turn_pol is not None:
        # 振り返る間だけ、回れ右の方策に替える
        tm_ = (stage == TURN_PROD) | ((stage == TURN_REJ) & a.turn_rej)
        if tm_.any():
          act_l = torch.where(tm_[:, None], turn_pol(obs["actor"]), act_l)
      om = obs
      if a.flat_scan:
        # 取り上げ・置くの方策（平らな床だけで学習）には、地形の高さの入力を平らに見せる（各台の中央値でそろえる。台の縁・段が入らないように）
        om = obs.clone() if hasattr(obs, "clone") else {k_: v.clone() for k_, v in obs.items()}
        for g_ in ("actor", "critic"):
          hs = om[g_][:, 93:280]
          om[g_][:, 93:280] = hs.median(dim=1, keepdim=True).values.expand_as(hs)
      act_m = manip(om)
      if os.environ.get("FG_DBG_OBS"):
        # 最初に置くへ入った台の、置くの最初の 15 歩の観測と行動（と入った瞬間の記録）を保存
        if "dbg_i" not in dir():
          dbg_i, dbg_obs = None, []
        fpl_ = (stage == PLACE) & (t - t_stage < 0.5 * dt)
        if dbg_i is None and bool(fpl_.any()):
          dbg_i = int(torch.nonzero(fpl_).flatten()[0])
        if dbg_i is not None and len(dbg_obs) < 40:
          dbg_obs.append((om["actor"][dbg_i].clone().cpu(), act_m[dbg_i].clone().cpu(), float(s["tilt"][dbg_i]), float(s["gap"][dbg_i])))
          if len(dbg_obs) == 40:
            torch.save({"traj": dbg_obs, "entry_place": [entry_place[dbg_i]]}, os.environ["FG_DBG_OBS"])
      # 置くに入った最初の 1 歩の観測を記録（学習の始めの場面との食いちがいを調べる：2026-10-13 16:00）
      first_pl = (stage == PLACE) & (t - t_stage < 0.5 * dt)
      for i in torch.nonzero(first_pl).flatten().tolist():
        if entry_place[i] is not None and "obs" not in entry_place[i]:
          entry_place[i]["obs"] = om["actor"][i].tolist()
      act = torch.where(manip_m[:, None], act_m, act_l)
      if a.manip_legs_loco:
        # 取り上げ・置くの間も、脚は run14（腕を動かされても立つ練習をした方策）、腕だけ V16
        arm_i = at._arm_idx
        act = act_l.clone()
        act[:, arm_i] = torch.where(manip_m[:, None], act_m[:, arm_i], act_l[:, arm_i])
    obs, _, dones, _ = wenv.step(act)
    if a.metrics:
      # 段階ごとの所要時間と滑らかさ（台数 × 段階）。加速度・躍度は関節速度の差分、仕事率は |アクチュエータの力 × 速度| の和
      if k == 0:
        mt_leg = torch.tensor([any(x in nm for x in ("hip", "knee", "ankle")) for nm in robot.joint_names], device=dev)
        mt_arm = torch.zeros(len(robot.joint_names), dtype=torch.bool, device=dev)
        mt_arm[env._fg_arm_jids] = True
        MT = {kk: torch.zeros(n, 12, device=dev) for kk in ("時間", "止まる", "加速度_脚", "加速度_腕", "躍度_脚", "躍度_腕", "仕事", "胴体の揺れ", "箱の加速度")}
        mt_qv = robot.data.joint_vel.clone()
        mt_qa = torch.zeros_like(mt_qv)
        mt_bv = box.data.root_link_lin_vel_w.clone()
      qv_ = robot.data.joint_vel
      qa_ = (qv_ - mt_qv) / dt
      jk_ = (qa_ - mt_qa) / dt
      bv_ = box.data.root_link_lin_vel_w
      ba_ = (torch.linalg.norm(bv_ - mt_bv, dim=-1) / dt).clamp(max=50.0)  # 箱を置き直した瞬間の跳びを抑える
      oh_ = torch.nn.functional.one_hot(stage.clamp(0, 11), 12).float() * ((stage < DONE) & (k >= 2))[:, None].float() * dt
      MT["時間"] += oh_
      MT["止まる"] += oh_ * ((torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=-1) < 0.05) & (robot.data.root_link_ang_vel_b[:, 2].abs() < 0.1)).float()[:, None]
      MT["加速度_脚"] += oh_ * qa_[:, mt_leg].square().mean(-1)[:, None]
      MT["加速度_腕"] += oh_ * qa_[:, mt_arm].square().mean(-1)[:, None]
      MT["躍度_脚"] += oh_ * jk_[:, mt_leg].square().mean(-1)[:, None]
      MT["躍度_腕"] += oh_ * jk_[:, mt_arm].square().mean(-1)[:, None]
      MT["仕事"] += oh_ * (env.sim.data.qfrc_actuator * env.sim.data.qvel).abs().sum(-1)[:, None]
      MT["胴体の揺れ"] += oh_ * robot.data.root_link_ang_vel_b[:, :2].square().sum(-1)[:, None]
      MT["箱の加速度"] += oh_ * ba_.square()[:, None]
      mt_qv, mt_qa, mt_bv = qv_.clone(), qa_, bv_.clone()
    newf = dones.bool() & ~fell & (stage != DONE) & (stage != FAIL)
    tm_ = env.termination_manager
    for i in torch.nonzero(newf).flatten().tolist():
      # どの終了の条件か（「転倒」は終了の全部をまとめていた。取り上げ単体では転倒 0〜0.6 % なのに統合で 15〜25 % の食いちがいを調べる）
      terms_ = [nm for nm in tm_.active_terms if bool(tm_.get_term(nm)[i])]
      fail_why[i] = f"{NAMES[int(stage[i])]}：転倒" if terms_ in (["fell_over"], []) else f"{NAMES[int(stage[i])]}：終了（{'・'.join(terms_)}）"
      log.append(f"  台{i} {NAMES[int(stage[i])]} 転倒：段階に入って {t - float(t_stage[i]):.1f} 秒")
    stage[newf] = FAIL
    fell |= newf
    if a.video and t < a.video_seconds:
      frames.append(env.render())
    if a.replay_video and k % 2 == 0:
      qhist.append(env.sim.data.qpos.clone())
    if k % 50 == 0:
      log.append(f"t={t:.0f} 台0 段階 {NAMES[int(stage[0])]} 往復 {int(trips[0])} 胴体 x {float(p[0, 0]):+.2f} z {float(p[0, 2]):+.2f} 向き {math.degrees(float(yaw[0])):+.0f}")
    newfail = (stage == FAIL) & torch.isnan(t_fail)
    t_fail[newfail] = t
    if bool(((stage == DONE) | (stage == FAIL)).all()):
      break
  ok = stage == DONE
  if a.metrics and ok.any():
    import json

    print("所要時間と滑らかさ（完了した台の平均。加速度・躍度・揺れ・箱は二乗平均の平方根）")
    print(f"  {'段階':28s} {'秒':>6s} {'止まる秒':>7s} {'脚 加速度':>8s} {'腕 加速度':>8s} {'脚 躍度':>8s} {'腕 躍度':>8s} {'仕事 J':>7s} {'揺れ':>6s} {'箱 加速':>6s}")
    res = {}
    for j in list(range(10)) + [None]:
      g_ = {kk: (v[ok, j] if j is not None else v[ok].sum(-1)) for kk, v in MT.items()}
      T_ = g_["時間"].clamp(min=1e-6)
      r_ = {"秒": g_["時間"].mean().item(), "止まる秒": g_["止まる"].mean().item(), "仕事J": g_["仕事"].mean().item()}
      for kk in ("加速度_脚", "加速度_腕", "躍度_脚", "躍度_腕", "胴体の揺れ", "箱の加速度"):
        r_[kk] = (g_[kk] / T_).sqrt().mean().item()
      nm_ = NAMES[j] if j is not None else "合計"
      res[nm_] = r_
      print(f"  {nm_:28s} {r_['秒']:6.1f} {r_['止まる秒']:7.1f} {r_['加速度_脚']:8.1f} {r_['加速度_腕']:8.1f} {r_['躍度_脚']:8.0f} {r_['躍度_腕']:8.0f} "
            f"{r_['仕事J']:7.0f} {r_['胴体の揺れ']:6.2f} {r_['箱の加速度']:6.2f}")
    Path(a.metrics).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  単位：秒、rad/s²、rad/s³、J、rad/s、m/s²")
  print(f"統合：{a.trips} 往復、{n} 台、方策 歩く={a.loco} 取り上げ・置く={a.manip}")
  print(f"  完了 {int(ok.sum())} 台（{ok.float().mean().item() * 100:.1f} %）" + (f"、平均 {torch.nanmean(t_done).item():.0f} 秒" if ok.any() else ""))
  print(f"  往復の数の分布：" + "、".join(f"{j} 往復 {int((trips == j).sum())} 台" for j in range(a.trips + 1)))
  from collections import Counter

  c = Counter(w for w, st in zip(fail_why, stage.tolist()) if st == FAIL)
  for w, v in c.most_common():
    print(f"  失敗：{w} {v} 台")
  print(f"  終わらず（時間切れ）：{int(((stage != DONE) & (stage != FAIL)).sum())} 台")
  # 理由ごとに最大 8 行（先頭 30 行だけでは転倒の行で埋まり、寄れない台の様子が見えなかった）
  sel_ = []
  for kw in ("取り上げ前の落下", "転倒", "寄れない", "終わらず", "下がれない", "置く：", "置く 落下", "取り上げ 落下", "（取り上げた後） 落下"):
    sel_ += [x for x in log if x.startswith("  台") and kw in x][:8]
  if os.environ.get("FG_DBG_TURN"):
    sel_ += [x for x in log if x.startswith("  回る")]
  for line in sel_ + log[-5:]:
    print("  " + line)
  if a.entry_log:
    torch.save({"entry_place": entry_place, "entry_place_all": entry_place_all, "place_trace": place_trace if (a.place_closed or a.place_v2) else None, "pick_trace": pick_trace, "appr_trace": appr_trace, "entry": entry, "fail_why": fail_why, "stage": stage.tolist(), "trips": trips.tolist()}, a.entry_log)
    print("入った瞬間の記録:", a.entry_log)
  if a.replay_video and qhist:
    done_ids = torch.nonzero(ok).flatten().tolist()
    why_sel = os.environ.get("FG_REPLAY_WHY", "")
    first = 0
    if why_sel:
      # 失敗の理由にこの語を含む台の、失敗の 25 秒前から 1 秒後まで
      done_ids = [j for j, w in enumerate(fail_why) if why_sel in w and int(stage[j]) == FAIL]
      pd_ = os.environ.get("FG_REPLAY_PICKDUR")
      if pd_ and done_ids:
        # 取り上げの長さがこの秒数に近い台から順に並べる（典型的な失敗を選ぶため：2026-10-18）
        def _pdur(j):
          tr_ = pick_trace[j] or []
          rr = [r for r in tr_ if r[0] == PICK and r[1] == tr_[-1][1]] if tr_ else []
          return rr[-1][2] if rr else -99.0
        done_ids.sort(key=lambda j: abs(_pdur(j) - float(pd_)))
        print("取り上げの長さ（近い順）:", [(j, _pdur(j)) for j in done_ids[:6]])
    if not done_ids:
      print("再生する台がない")
    else:
      import imageio.v3 as iio

      i = done_ids[int(os.environ.get("FG_REPLAY_K", "0")) % len(done_ids)]
      env._offline_renderer._env_ids = (i,)
      if why_sel:
        first = max(0, int((float(t_fail[i]) - float(os.environ.get("FG_REPLAY_PRE", "25"))) / dt / 2))
        last = int(float(t_fail[i]) / dt / 2) + 25
        print(f"再生：台 {i}「{fail_why[i]}」、失敗 {float(t_fail[i]):.1f} 秒")
      else:
        last = int(float(t_done[i]) / dt / 2) + 50
        if os.environ.get("FG_REPLAY_TURN"):
          # 置いた後の振り返りだけ（入る 1 秒前から、終わって 1 秒後まで）。完了した台の t_stage は振り返りに入った時刻のまま
          first = max(0, int((float(t_stage[i]) - 1.0) / dt / 2))
          last = int(float(t_done[i]) / dt / 2) + 25
          print(f"再生：台 {i} の振り返り {float(t_stage[i]):.1f}〜{float(t_done[i]):.1f} 秒（{float(t_done[i]) - float(t_stage[i]):.1f} 秒）")
      # 1 枚ずつ書き出す（3 往復で約 6000 枚、全部を溜めると約 16 GB になり、メモリ不足で止められた：2026-10-16）
      import imageio.v2 as iio2

      # FG_DUMP_SCENE=<.npz>：Blender へ渡すため、見えている形状ごとの位置・向き（毎コマ）と、メッシュの頂点・面を書き出す（2026-10-19）
      dump = os.environ.get("FG_DUMP_SCENE")
      import mujoco

      rd = env._offline_renderer
      gpos, gquat, gids = [], [], None
      with iio2.get_writer(a.replay_video, fps=int(round(1 / dt / 2)), codec="libx264", macro_block_size=8) as wr:
        for q in qhist[first:last]:
          env.sim.data.qpos[:] = q
          rd.update(env.sim.data)
          wr.append_data(rd.render())
          if dump:
            m_, d_ = rd._model, rd._data
            if gids is None:
              gids = np.nonzero((m_.geom_rgba[:, 3] > 0.01) & (m_.geom_group <= 2))[0]
            q_ = np.zeros((len(gids), 4))
            for k_, g_ in enumerate(gids):
              mujoco.mju_mat2Quat(q_[k_], d_.geom_xmat[g_])
            gpos.append(d_.geom_xpos[gids].copy())
            gquat.append(q_)
      print(f"再生の動画: {a.replay_video}（台 {i}）")
      if dump and gids is not None:
        m_ = rd._model
        meshes = {}
        for g_ in gids:
          mid = int(m_.geom_dataid[g_])
          if m_.geom_type[g_] == mujoco.mjtGeom.mjGEOM_MESH and mid >= 0 and mid not in meshes:
            v0, nv = m_.mesh_vertadr[mid], m_.mesh_vertnum[mid]
            f0, nf = m_.mesh_faceadr[mid], m_.mesh_facenum[mid]
            meshes[mid] = (m_.mesh_vert[v0:v0 + nv].astype(np.float32), m_.mesh_face[f0:f0 + nf].astype(np.int32))
        names = [mujoco.mj_id2name(m_, mujoco.mjtObj.mjOBJ_GEOM, int(g_)) or f"geom{int(g_)}" for g_ in gids]
        np.savez_compressed(dump, fps=1 / dt / 2, names=np.array(names), type=m_.geom_type[gids], size=m_.geom_size[gids],
                            rgba=m_.geom_rgba[gids], dataid=m_.geom_dataid[gids], pos=np.array(gpos, np.float32), quat=np.array(gquat, np.float32),
                            **{f"mesh{k}_v": v for k, (v, f) in meshes.items()}, **{f"mesh{k}_f": f for k, (v, f) in meshes.items()})
        print(f"Blender 用の書き出し: {dump}（形状 {len(gids)}、コマ {len(gpos)}、メッシュ {len(meshes)}）")
  if a.video and frames:
    import imageio.v3 as iio

    iio.imwrite(a.video, np.stack(frames), fps=int(round(1 / dt)))
    print("動画:", a.video)


if __name__ == "__main__":
  main()
