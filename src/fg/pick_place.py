"""排斥用テーブルから回収箱を取り上げ、約 2.5 m 歩いて、検査用テーブルに置く（mjlab 1.6.0）。

脚・腰は学習した歩行の方策（落ちる箱を持つ課題 FG-Carry-Steps-Luna-V11）。腕は逆運動学（fg.arm_ik）で動かす。
上位の手順（状態機械）と、歩く目標への速度の指令はこのスクリプトが出す（計画書 10 章の「上位の状態管理と低レベルの運動制御の分離」）。

手順（1 台ごとに進む）
  0 排斥用テーブルへ近づく（腕は手を 10 cm 外へ開いた構え）
  1 そろえる：箱の実際の位置・向き（上体から見た）に合わせ、受け板をフランジの 1 cm 下・外へ 10 cm の位置へ
  2 差し込む：手を内へ 10 cm 寄せる（外側の縁がフランジの縁の外に来る）
  3 持ち上げる：3 cm 上げる（箱がテーブルから 2 cm 浮く）
  4 後ろへ下がる（0.4 m）
  5 腕を抱える姿勢へ戻す
  6 その場で 180 度回る
  7 検査用テーブルへ近づく
  8 下ろす：箱がテーブルに着き、受け板がフランジの 4 mm 下に来るまで下げる（下げる量は、そのときの箱の高さから決める）
  9 手を外へ 10 cm 開く
  10 後ろへ下がる（0.4 m）→ 完了

知覚は Phase 1 の扱い：テーブル・箱の位置はシミュレーターの正確な値を使う（実機では頭のカメラで測る：Phase 2・3）

使い方（リポジトリの直下で）:
  python src/fg/pick_place.py <checkpoint> --num-envs 64 [--video out.mp4]
"""

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fg.carry_steps  # noqa: E402,F401
from fg.arm_ik import ArmIK  # noqa: E402
from fg.free_bin import HOOK_PLATE_T, layout  # noqa: E402
from fg.luna_constants import BIN_FLANGE_T, BIN_OUTER  # noqa: E402
from mjlab.entity import EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs import mdp as envs_mdp  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply  # noqa: E402

TASK = "FG-Carry-Steps-Luna-V11"

# 配置（各台の区画の座標。ロボットは原点から +x 向きにスタート）
XA = 1.2  # 排斥用テーブルの中心（箱の中心）
XB = XA - 2.5  # 検査用テーブル（2.5 m 離す。ロボットは 180 度回って歩く）

# テーブルの寸法 [m]（Luna に合わせて設計。高さは測った値から決める：main の --zc）
TOP_X, TOP_Y, TOP_T = 0.28, 0.20, 0.03  # 天板：左右は腕（手首は左右 ±0.18 m）に当たらない幅。前後は、置くときに胴体が 7〜8 cm 前後にずれても箱が載るよう、箱の底（173 mm）より長く（16 → 28 cm、2026-10-12 13 時）
POST = 0.06  # 脚（中央の 1 本）の太さ
PICK_CLEAR = 0.010  # 取り上げ：差し込むときの受け板の上面とフランジの下面のすき間。
# 受け具の溝（受け板と押さえの間）は 8 mm。手が高すぎると受け板の先がフランジの縁に当たって止まるので、低めに構え、
# 押さえの内側の斜面（15 mm）でフランジを溝へ案内する（手の高さの許容：高い側 約 10 mm、低い側 約 13 mm）
LIFT = 0.03  # 取り上げで上げる量（箱がテーブルから 2 cm 浮く）
PLACE_CLEAR = 0.005  # 置くとき、腕を抱える姿勢より少しだけ下げる（上げて近づき、抱える姿勢へ戻して載せる）。抱える姿勢より 2 cm 以上下げると、
# 方策が上体を後ろへ反らせて（最大 −22 度）手が上がり、箱が 12 度傾いた（run14、2026-10-12 13 時）
PLACE_REST = 0.012  # 箱の底と天板の差がこれ以下なら手を開く（1 cm ほど落としても缶はこぼれない）。大きければ 5 mm ずつ追加で下げる（3 回まで）
HOVER = 0.035  # 抱えたときの箱の底（平均）と検査用テーブルの天板のすき間（立っているときの箱の高さは台ごとに ±3〜4 cm ちがう）
PLATE_Z = BIN_OUTER[2] / 2 - BIN_FLANGE_T - HOOK_PLATE_T / 2  # 抱えたときの受け板の中心の高さ（箱の座標）
PLATE_Y = 0.146  # 抱えたときの受け板の中心の左右の位置（箱の座標、絶対値。受け板の内側 0.129〜外側 0.163）
SQUEEZE = 0.008  # 差し込むとき、手を抱える姿勢より内へ寄せる量。外側の縁がフランジの縁を押して箱を真ん中にそろえる
# （寄せないと、手が 1 cm ほど開いたままで、受け板がフランジ（幅 22 mm）に 5 mm しかかからない側があった）
SPREAD = 0.10  # 差し込む前・置いたあとに手を外へ開く量（3・6・8 cm では、近づくときの横・向きのずれで受け具が箱の壁に当たった）

TABLE_RGBA = (0.55, 0.55, 0.58, 1.0)


def table_spec(top_z: float, name: str, top_x: float = TOP_X, top_y: float = TOP_Y):
  def fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name=name)
    for nm, size, pos in (
      ("top", (top_x / 2, top_y / 2, TOP_T / 2), (0.0, 0.0, top_z - TOP_T / 2)),
      ("post", (POST / 2, POST / 2, (top_z - TOP_T) / 2), (0.0, 0.0, (top_z - TOP_T) / 2)),
    ):
      for kind in ("collision", "visual"):
        b.add_geom(name=f"{name}_{nm}_{kind}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos),
                   rgba=list(TABLE_RGBA), group=3 if kind == "collision" else 2,
                   contype=1 if kind == "collision" else 0, conaffinity=1 if kind == "collision" else 0,
                   condim=3 if kind == "collision" else 1)
    return spec

  return fn


def reset_bin_on_table(env, env_ids, top_z: float):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  box = env.scene["bin"]
  oz = BIN_OUTER[2] / 2
  p = env.scene.env_origins[env_ids] + torch.tensor([XA, 0.0, top_z + oz + 0.001], device=env.device)
  q = torch.tensor([1.0, 0, 0, 0], device=env.device).expand(len(env_ids), 4)
  box.write_root_link_pose_to_sim(torch.cat([p, q], dim=-1), env_ids=env_ids)
  box.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def make_env_cfg(num_envs: int, zc: float, task: str = TASK, start_x: float = 0.0, carrying: bool = False,
                 nominal: bool = False, hover: float | None = None, pick_offset: float = 0.0):
  """carrying=True：箱を抱えて、検査用テーブルの方を向いて start_x から始める（置く部分だけを試す）"""
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.events.pop("randomize_terrain", None)
  if nominal:
    # 設計どおりのロボット（関節のエンコーダーのずれ ±0.015 rad・重心のずれの乱数を切る）。乱数ありでは、抱えた箱の高さが
    # 台ごとに 931〜959 mm ちがい（設計どおりなら 945.7〜946.6 mm）、腕だけでは合わせきれない（run14、2026-10-12 13 時）
    for nm in ("encoder_bias", "base_com"):
      cfg.events.pop(nm, None)
  yaw0 = math.pi if carrying else 0.0
  cfg.events["reset_base"].params["pose_range"] = {"x": (start_x, start_x), "y": (0.0, 0.0), "yaw": (yaw0, yaw0)}
  cfg.terminations.pop("bin_dropped", None)  # 箱はテーブルの上から始まる。落下はこのスクリプトで数える
  cfg.events.pop("carry_mode", None)  # 学習用（箱を持つ・持たないの切り替え）
  for nm in ("bin_drop", "bin_tilt", "bin_offset"):
    cfg.rewards.pop(nm, None)  # 学習用の罰（評価には要らない）
  cfg.rewards.pop("ang_vel_error", None)
  cfg.episode_length_s = 1e9
  ha, hb = zc + 0.010 + pick_offset, zc - (HOVER if hover is None else hover)  # 排斥用は抱えた箱の底 ＋ 1 cm ＋ 補正（差し込みのすき間を変えてもテーブルは動かさない）
  # 排斥用テーブル：抱えた受け具の高さで、フランジとのすき間が PICK_CLEAR になる高さ
  # 検査用テーブル：抱えた箱の底より HOVER 低い
  cfg.scene.entities["table_a"] = EntityCfg(spec_fn=table_spec(ha, "table_a"), init_state=EntityCfg.InitialStateCfg(pos=(XA, 0.0, 0.0)))
  cfg.scene.entities["table_b"] = EntityCfg(spec_fn=table_spec(hb, "table_b"), init_state=EntityCfg.InitialStateCfg(pos=(XB, 0.0, 0.0)))
  for nm in ("table_a", "table_b"):
    cfg.events[f"reset_{nm}"] = EventTermCfg(func=envs_mdp.reset_root_state_uniform, mode="reset",
                                             params={"pose_range": {}, "asset_cfg": SceneEntityCfg(nm)})
  if not carrying:
    cfg.events["reset_bin"] = EventTermCfg(func=reset_bin_on_table, mode="reset", params={"top_z": ha})
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.heading_command = False
  tw.ranges.heading = None
  tw.rel_standing_envs = 0.0
  tw.rel_heading_envs = 0.0
  tw.ranges.lin_vel_x = (0.0, 0.0)
  tw.ranges.lin_vel_y = (0.0, 0.0)
  tw.ranges.ang_vel_z = (0.0, 0.0)
  return cfg, ha, hb


def yaw_of(q: torch.Tensor) -> torch.Tensor:
  return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def wrap(a: torch.Tensor) -> torch.Tensor:
  return (a + math.pi) % (2 * math.pi) - math.pi


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--num-envs", type=int, default=16)
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--start-at-table", action="store_true", help="排斥用テーブルの 0.3 m 手前から始める（取り上げの部分だけを試す）")
  ap.add_argument("--start-carrying", action="store_true", help="箱を抱えて、検査用テーブルの 0.8 m 手前から始める（置く部分だけを試す）")
  ap.add_argument("--nominal", action="store_true", help="設計どおりのロボット（ばらつきの乱数を切る）")
  ap.add_argument("--hover", type=float, default=None, help="抱えた箱の底と検査用テーブルの天板のすき間 [m]（既定 HOVER）")
  ap.add_argument("--place-raise", type=float, default=0.03, help="検査用テーブルへ近づくとき、腕で箱を上げておく量 [m]（上体の座標）")
  ap.add_argument("--roll-comp", action="store_true", help="上体の左右の傾きを腕で補う（48 台で 16.7 → 6.2 % に下がったので既定は使わない）")
  ap.add_argument("--lift-check", action="store_true", help="持ち上がったか（高さ・傾き）を確かめ、上がらなければ追加で上げ、最後は失敗として止まる")
  ap.add_argument("--lift", type=float, default=None, help="持ち上げる量（腕の目標、上体の座標）[m]")
  ap.add_argument("--pick-yaw-tol", type=float, default=0.035, help="取り上げで近づくときの向きの許容 [rad]")
  ap.add_argument("--pick-retries", type=int, default=2, help="取り上げのやり直しの回数（下がれないとき、放して 0.25 m 下がり、近づくところから）")
  ap.add_argument("--face-bin", action="store_true", help="取り上げで近づくとき、箱の向きに合わせる（既定は向き 0）")
  ap.add_argument("--seat-check", action="store_true", help="持ち上げたあと受け板がフランジの下に入っているか確かめ、入っていなければ戻してそろえ直す")
  ap.add_argument("--plate-loop", action="store_true", help="差し込む前に受け板の高さを測って補正する（試したが、測るたびに 1 cm ほどばらつき、構えを変えると補正がずれて効かなかった）")
  ap.add_argument("--pick-clear", type=float, default=None, help="差し込むときの受け板の上面とフランジの下面のすき間 [m]（既定 PICK_CLEAR）")
  ap.add_argument("--squeeze", type=float, default=None, help="差し込むとき手を内へ寄せる量 [m]（既定 SQUEEZE）")
  ap.add_argument("--lock-waist-pick", action="store_true", help="取り上げの間、腰の yaw・roll を 0 に固定する（48 台で 16.7 → 10.4 % に下がったので既定は使わない）")
  ap.add_argument("--lock-waist", action="store_true", help="下ろす・手を開く間（段階 8・9）、腰の前後の傾きの目標を 0 に固定する")
  ap.add_argument("--pick-offset", type=float, default=0.0, help="排斥用テーブルの高さの補正 [m]")
  ap.add_argument("--tol", type=float, default=0.03, help="受け具を箱に寄せるときの前後の許容 [m]（横は 0.04 m）")
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--zc", type=float, default=None, help="抱えた箱の底の高さ [m]（省略すると最初に測る）")
  ap.add_argument("--video", type=str, default=None)
  ap.add_argument("--env-idx", type=int, default=0, help="動画に撮る台")
  ap.add_argument("--cam", type=float, nargs=3, default=(3.5, -20.0, 135.0), help="カメラの距離 [m]・仰角・方位 [度]")
  ap.add_argument("--hold", type=float, default=0.0, help="腕を動かす間、胴体を保つ指令の不感帯 [m]（0 なら保たない。run13b では 0.03 で揺れて転んだ）")
  ap.add_argument("--debug-contacts", action="store_true", help="終わりに、その台のテーブル・箱との接触を表示")
  args = ap.parse_args()
  global PICK_CLEAR, SQUEEZE, LIFT
  if args.lift is not None:
    LIFT = args.lift
  if args.pick_clear is not None:
    PICK_CLEAR = args.pick_clear
  if args.squeeze is not None:
    SQUEEZE = args.squeeze
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  zc = args.zc if args.zc is not None else 0.94
  sx = XB + 0.36 + 0.80 if args.start_carrying else (XA - 0.36 - 0.30 if args.start_at_table else 0.0)
  cfg, ha, hb = make_env_cfg(args.num_envs, zc, args.task, sx, carrying=args.start_carrying,
                             nominal=args.nominal, hover=args.hover, pick_offset=args.pick_offset)
  if args.video:
    cfg.viewer.env_idx = args.env_idx
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = args.cam
    cfg.viewer.width, cfg.viewer.height = 1280, 720
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if args.video else None)
  agent_cfg = load_rl_cfg(args.task)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(wenv, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  n = args.num_envs
  robot, box = env.scene["robot"], env.scene["bin"]
  act = env.action_manager.get_term("joint_pos")
  cmd = env.command_manager.get_term("twist")
  ik = ArmIK()
  names = act.arm_names

  def arm_vec(**kw):
    r, err, _ = ik.solve(**kw)
    return torch.tensor([r[nm] for nm in names], device=device), err

  carry_q, _ = arm_vec()
  # 近づくときの構え：手を外へ開き、さらに箱の底より下へ（歩くと手が左右に ±5〜10 cm 揺れ、受け具が箱の壁・フランジに当たったため）。
  # 箱より上に構える案は、歩くと上体が前に傾いて手が 8 cm 下がり、届かなかった。テーブルの天板は細い（左右 ±0.10 m）ので、
  # 下に構えた受け具（左右 ±0.19 m より外）は天板にも当たらない。手は前後に 19 度傾くが、構えでは何にも触れない
  RAISE_DX, RAISE_DZ = 0.0, 0.0
  # → これも、止まっているときは手が上がって受け具が箱の壁の高さに来た。最終的に：高さは差し込む高さのまま、手を外へ 10 cm 開く
  #   （受け具の内側の端が箱のフランジの縁から 6 cm 外。歩くときの手の横の揺れは ±5 cm ほど）
  spread_q, _ = arm_vec(dx=RAISE_DX, dz=RAISE_DZ, spread=SPREAD)
  torso_id = robot.find_bodies(("waist_pitch_link",))[0][0]
  wrist_ids = robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"))[0]
  Lo = layout()
  mid0 = (Lo["data"].xpos[Lo["lw"]] + Lo["data"].xpos[Lo["rw"]]) / 2
  off_mid = torch.tensor(Lo["center_w"] - mid0, dtype=torch.float32, device=device)  # 両手首の中点 → 抱えた箱の中心（向き 0 のとき）
  center_root = torch.tensor(layout()["center_root"], dtype=torch.float32, device=device)
  bin_t0 = torch.tensor(ik.bin_t, dtype=torch.float32, device=device)

  # 腕の目標の補間（台ごと）
  q_from = carry_q.expand(n, -1).clone()
  q_to = spread_q.expand(n, -1).clone()
  t_from = torch.zeros(n, device=device)
  dur = torch.ones(n, device=device)
  if args.start_carrying:
    q_to = carry_q.expand(n, -1).clone()
  act.arm_target = q_to.clone()
  act.arm_target_mask = torch.ones(n, dtype=torch.bool, device=device)
  act.schedule = False  # 課題 V14 の、学習用の腕の動き（このスクリプトで腕を動かす）

  state = torch.full((n,), 7 if args.start_carrying else 0, dtype=torch.long, device=device)
  t_state = torch.zeros(n, device=device)
  hold = torch.zeros(n, device=device)
  near = torch.zeros(n, dtype=torch.bool, device=device)  # 近づく 2 段階目（まっすぐ寄せる）に入ったか
  back_from = torch.zeros(n, 2, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  dropped = torch.zeros(n, dtype=torch.bool, device=device)
  t_done = torch.full((n,), float("nan"), device=device)
  t_pick = torch.full((n,), float("nan"), device=device)
  origin = env.scene.env_origins
  frames = []
  drop_info = []  # 箱を落としたときの段階
  fall_info = []  # 転んだ台・時刻・段階・そのとき箱を落としていたか
  log = []

  def set_arm(mask, goal, t, d):
    nonlocal q_from, q_to
    cur = act.arm_target.clone()
    q_from = torch.where(mask[:, None], cur, q_from)
    q_to = torch.where(mask[:, None], goal, q_to)
    t_from[mask] = t
    dur[mask] = d

  def go_state(mask, s, t):
    near[mask] = False
    state[mask] = s
    t_state[mask] = t
    hold[mask] = 0.0

  hold_xy = torch.zeros(n, 2, device=device)
  side_dz = torch.zeros(n, 2, device=device)  # 受け板の高さの補正（右・左）[m]
  pick_dz0 = torch.zeros(n, device=device)  # そろえ始めに測った、受け具を合わせる高さ（上体の座標）[m]
  plate_fix = torch.zeros(n, dtype=torch.long, device=device)  # 受け板の高さを補正した回数
  plate_err = torch.full((n,), 1.0, device=device)  # 受け板の高さの差（左右の大きい方）[m]
  hook_gid = [env.sim.mj_model.geom(f"robot/{sd}_hook_plate_collision").id for sd in ("right", "left")]
  place_dz = torch.zeros(n, device=device)
  place_try = torch.zeros(n, dtype=torch.long, device=device)
  arm_up = torch.zeros(n, dtype=torch.bool, device=device)
  lift_try = torch.zeros(n, dtype=torch.long, device=device)
  raise_q, _ = arm_vec(dz=args.place_raise)
  open_up_q, _ = arm_vec(dz=0.02, spread=SPREAD)
  open_up2_q, _ = arm_vec(dz=0.04, spread=SPREAD)
  retry_pick = torch.zeros(n, dtype=torch.long, device=device)
  retry_back = torch.zeros(n, dtype=torch.long, device=device)
  unhook14 = torch.zeros(n, dtype=torch.bool, device=device)
  hold_yaw = torch.zeros(n, device=device)

  def pick_goal(i, lift, spread, log_it=False, fresh=False):
    """上体から見た箱の位置・向きを測り、受け具をフランジの下（＋lift）へ合わせる腕の目標を解く。"""
    tp = robot.data.body_link_pos_w[i, torso_id]
    tq = robot.data.body_link_quat_w[i:i + 1, torso_id]
    qi = tq * torch.tensor([1.0, -1.0, -1.0, -1.0], device=device)
    rel = quat_apply(qi, (box.data.root_link_pos_w[i] - tp)[None])[0] - bin_t0
    dyaw = float(wrap(yaw_of(box.data.root_link_quat_w[i:i + 1]) - yaw_of(tq))[0])
    # 箱の左右の傾き（上体から見て）：上体が左右に 2 度傾くと、左右の受け具の高さが 1.4 cm ちがい、片方がフランジの上に乗った（run14）
    ey_b = quat_apply(qi, quat_apply(box.data.root_link_quat_w[i:i + 1], torch.tensor([[0.0, 1.0, 0.0]], device=device)))[0]
    droll = float(torch.atan2(ey_b[2], ey_b[1])) if args.roll_comp else 0.0
    # 左右は ±2.5 cm まで腕で合わせる（肩の roll の可動範囲の限界。残りは外側の縁が箱を押して、テーブルの上で滑らせてそろえる）
    dx_, dy_ = float(rel[0]), float(max(-0.025, min(0.025, rel[1])))
    # 高さは、そろえ始めに測った値に固定する（測り直すと、腕を動かす→方策が上体の姿勢を変える→測った高さが変わる→腕を動かす、
    # の繰り返しで受け板が ±3 cm 揺れた。止まっているときの方策そのものの揺れは手首で 1.8 mm（measure_sway.py））。
    # 高さの微調整は、受け板の位置を直接測る閉ループ（side_dz）で行う
    if fresh or not args.plate_loop:
      pick_dz0[i] = float(rel[2]) - PICK_CLEAR
    dz_ = float(pick_dz0[i]) + lift
    so = {"right": (0.0, 0.0, float(side_dz[i, 0])), "left": (0.0, 0.0, float(side_dz[i, 1]))}
    g, err = arm_vec(dx=dx_, dy=dy_, dz=dz_, dyaw=dyaw, spread=spread, droll=droll, side_off=so)
    if log_it:
      log.append(f"t={t:.1f} 台 {i}：箱のずれ（上体から）x {rel[0] * 100:+.1f} y {rel[1] * 100:+.1f} z {rel[2] * 100:+.1f} cm、向き {math.degrees(dyaw):+.1f} 度、左右の傾き {math.degrees(droll):+.1f} 度、逆運動学の誤差 {err * 1000:.1f} mm")
    return g, err

  def hold_base(mask):
    """胴体を記録した位置・向きに保つ速度の指令（3 cm・3 度の不感帯。方策が足を出す最低の大きさで）。"""
    nonlocal vx, vy, wz
    if args.hold <= 0:
      return
    db = args.hold
    e = hold_xy - p[:, :2]
    ex = torch.cos(yaw) * e[:, 0] + torch.sin(yaw) * e[:, 1]
    ey = -torch.sin(yaw) * e[:, 0] + torch.cos(yaw) * e[:, 1]
    eyaw = wrap(hold_yaw - yaw)
    hx = torch.where(ex.abs() > db, torch.sign(ex) * 0.12, torch.zeros_like(ex))
    hy = torch.where(ey.abs() > db, torch.sign(ey) * 0.15, torch.zeros_like(ey))
    hw = torch.where(eyaw.abs() > 0.05, torch.sign(eyaw) * 0.25, torch.zeros_like(eyaw))
    vx = torch.where(mask, hx, vx)
    vy = torch.where(mask, hy, vy)
    wz = torch.where(mask, hw, wz)

  obs = wenv.get_observations()
  dt = env.step_dt
  steps = int(args.seconds / dt)
  for k in range(steps):
    t = k * dt
    # 腕の補間（なめらかに：3 次）
    a = torch.clamp((t - t_from) / dur, 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    act.arm_target = q_from + (q_to - q_from) * a[:, None]

    p = robot.data.root_link_pos_w - origin
    yaw = yaw_of(robot.data.root_link_quat_w)
    # 受け具の位置（抱えたときの箱の中心が来る所）：両手首の中点＋向きで回した決まったずれ。
    # 胴体の位置から決めると、上体の前傾で手が数 cm 前・下にあるぶんずれて、受け具が箱の壁に当たった
    mid = robot.data.body_link_pos_w[:, wrist_ids].mean(dim=1) - origin
    c_, s_ = torch.cos(yaw), torch.sin(yaw)
    pocket = mid[:, :2] + torch.stack([c_ * off_mid[0] - s_ * off_mid[1], s_ * off_mid[0] + c_ * off_mid[1]], dim=-1)
    # 構え（前後に RAISE_DX）のぶんを戻す：抱える姿勢に戻したときの受け具の位置
    raised = (state == 0)[:, None]
    pocket = pocket - raised * RAISE_DX * torch.stack([c_, s_], dim=-1)
    vx = torch.zeros(n, device=device)
    vy = torch.zeros(n, device=device)
    wz = torch.zeros(n, device=device)

    def approach(goal_xy, goal_yaw, tol, mask, tol_y=0.04, tol_yaw=0.07):
      """目標の胴体の位置・向きへの速度の指令（2 段階）。

      1) 手前の点（目標の 0.30 m 手前、向きは目標と同じ）へ：比例で最大 0.3 m/s。横 3 cm・向き 3 度以内で 2) へ
      2) まっすぐ寄せる：前後・左右とも、誤差が tol を超える向きに 0.12 m/s、tol 以内なら止まる
      （方策は 0.06 m/s のような小さな指令では足を出さない。比例のままだと 10 cm 行き過ぎ、横に 10 cm ずれて受け具が箱に当たった）
      """
      fwd = torch.stack([torch.cos(goal_yaw), torch.sin(goal_yaw)], dim=-1)
      tgt = torch.where(near[:, None], goal_xy, goal_xy - 0.30 * fwd)
      e = tgt - pocket
      ex = torch.cos(yaw) * e[:, 0] + torch.sin(yaw) * e[:, 1]
      ey = -torch.sin(yaw) * e[:, 0] + torch.cos(yaw) * e[:, 1]
      eyaw = wrap(goal_yaw - yaw)
      creep = 0.12
      # 遠くでも、残りが小さいと比例の指令が 0.07 m/s ほどになり、方策が足を出さずに止まった（run13b でも）→ 最低 0.15 m/s
      vx_far = torch.where(ex.abs() > 0.02, torch.sign(ex) * torch.clamp(0.8 * ex.abs(), 0.15, 0.3), torch.zeros_like(ex))
      vx_far = torch.clamp(vx_far, -0.2, 0.3)
      vy_far = torch.where(ey.abs() > 0.02, torch.sign(ey) * torch.clamp(0.8 * ey.abs(), 0.12, 0.15), torch.zeros_like(ey))
      # 横・向きの残りは腕（逆運動学）で合わせるので、許容を広めに（方策は止まっているとき 0.15 m/s の横の指令では動かなかった）
      vx_near = torch.where(ex.abs() > tol, torch.sign(ex) * creep, torch.zeros_like(ex))
      vy_near = torch.where(ey.abs() > tol_y, torch.sign(ey) * 0.15, torch.zeros_like(ey))
      # 向き：小さな旋回の指令（0.1 rad/s）では方策が回らず、6 度ずれたままだった → 最低 0.25 rad/s
      wz_cmd = torch.where(eyaw.abs() > min(0.05, tol_yaw * 0.8), torch.sign(eyaw) * torch.clamp(1.0 * eyaw.abs(), 0.25, 0.4), torch.zeros_like(eyaw))
      # 向きが 6 度以上ずれたら、進まずに向きだけ直す（歩きながらだと向きが 16〜26 度ずれ、受け具が箱に当たった）
      turn_only = eyaw.abs() > 0.10
      vx_far = torch.where(turn_only, torch.zeros_like(vx_far), vx_far)
      vy_far = torch.where(turn_only, torch.zeros_like(vy_far), vy_far)
      vx_near = torch.where(turn_only, torch.zeros_like(vx_near), vx_near)
      vy_near = torch.where(turn_only, torch.zeros_like(vy_near), vy_near)
      pre_ok = (torch.linalg.norm(e, dim=-1) < 0.04) & (ey.abs() < 0.03) & (eyaw.abs() < 0.05)
      near[mask & ~near & pre_ok] = True
      ok = near & (ex.abs() <= tol) & (ey.abs() <= tol_y) & (eyaw.abs() < tol_yaw)
      return torch.where(near, vx_near, vx_far), torch.where(near, vy_near, vy_far), wz_cmd, ok

    # 0 排斥用テーブルへ：箱が抱える位置に来る胴体の位置（向き 0）
    m0 = state == 0
    if m0.any():
      goal = (box.data.root_link_pos_w - origin)[:, :2]  # 受け具を箱の位置へ
      # 向きは 2 度以内（手を開いた構えは、向きが 6 度ずれると逆運動学の誤差が 2〜4 cm になり、受け具の高さが左右で 2 cm ちがった）
      # 向きは箱の向きに合わせる（やり直しのとき、箱がテーブルの上で回っていて、向き 0 のままでは近づけずに止まった：96 台で 19 台）
      gyaw = yaw_of(box.data.root_link_quat_w) if args.face_bin else torch.zeros(n, device=device)
      cx, cy, cw, ok = approach(goal, gyaw, args.tol, m0, tol_yaw=args.pick_yaw_tol)
      vx = torch.where(m0, cx, vx)
      vy = torch.where(m0, cy, vy)
      wz = torch.where(m0, cw, wz)
      hold = torch.where(m0 & ok, hold + dt, torch.where(m0, torch.zeros_like(hold), hold))
      nxt = m0 & (hold > 0.5)
      if nxt.any():
        # 1 そろえる：上体から見た箱の位置・向きを測り、逆運動学で受け具を合わせる
        side_dz[nxt] = 0.0
        plate_fix[nxt] = 0
        plate_err[nxt] = 1.0
        hold_xy[nxt] = p[nxt, :2]
        hold_yaw[nxt] = yaw[nxt]
        for i in torch.nonzero(nxt).flatten().tolist():
          g, err = pick_goal(i, lift=0.0, spread=SPREAD, log_it=i == args.env_idx, fresh=True)
          mk = torch.zeros(n, dtype=torch.bool, device=device)
          mk[i] = True
          set_arm(mk, g.expand(n, -1), t, 1.5)
        go_state(nxt, 1, t)

    # 1→2→3：そろえる → 差し込む → 持ち上げる（腕の補間が終わったら次へ）
    # 1→2→3：そろえる → 差し込む → 持ち上げる（段階は後ろから判定する。前から判定すると、同じステップで 1→2→3→4 と
    # 一気に進み、差し込む前に下がっていた。2026-10-12 昼に発見）
    # 腕を動かすと、方策は指令 0 でも 10 cm ほど後ろへ下がった → 胴体の位置を保つ指令を出しつつ、0.25 秒ごとに
    # 上体から見た箱の位置を測り直して、腕の目標を解き直す（閉ループ）
    arm_done = t >= t_from + dur + 0.3
    m3 = (state == 3) & arm_done
    if args.seat_check:
      # 持ち上げたあと、左右の受け板がフランジの下に正しく入っているかを、箱の座標で確かめる（Phase 1：シミュレーターの正確な値）。
      # 入っていなければ下がらずに（下がると箱をテーブルから引きずり落とした：192 台で 43 台）、箱をテーブルへ戻して手を開き（段階 15）、その場でそろえ直す
      gx = env.sim.data.geom_xpos[:, hook_gid]
      bq_i = box.data.root_link_quat_w * torch.tensor([1.0, -1, -1, -1], device=device)
      rel_p = quat_apply(bq_i[:, None, :].expand(n, 2, 4).reshape(-1, 4), (gx - box.data.root_link_pos_w[:, None, :]).float().reshape(-1, 3)).reshape(n, 2, 3)
      seated = (((rel_p[:, :, 2] - PLATE_Z).abs() < 0.008) & ((rel_p[:, :, 1].abs() - PLATE_Y).abs() < 0.012)).all(dim=-1)
      bad = m3 & ~seated & (retry_pick < args.pick_retries)
      for i in torch.nonzero(bad).flatten().tolist():
        retry_pick[i] += 1
        g, _ = pick_goal(i, lift=0.0, spread=-SQUEEZE)
        mk = torch.zeros(n, dtype=torch.bool, device=device)
        mk[i] = True
        set_arm(mk, g.expand(n, -1), t, 1.0)
        if i == args.env_idx:
          log.append(f"t={t:.1f} 台 {i}：受け板がフランジの下に入っていない（右 y {float(rel_p[i, 0, 1]):+.3f} z {float(rel_p[i, 0, 2]):+.3f}、左 y {float(rel_p[i, 1, 1]):+.3f} z {float(rel_p[i, 1, 2]):+.3f}）→ 戻してそろえ直す")
      go_state(bad, 15, t)
      m3 = m3 & ~bad
    # 15 箱をテーブルへ戻した → 16 手を開く → そろえ直す（段階 1）
    n15 = (state == 15) & arm_done
    for i in torch.nonzero(n15).flatten().tolist():
      g, _ = pick_goal(i, lift=0.0, spread=SPREAD)
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      set_arm(mk, g.expand(n, -1), t, 1.0)
    go_state(n15, 16, t)
    n16 = (state == 16) & arm_done
    for i in torch.nonzero(n16).flatten().tolist():
      g, _ = pick_goal(i, lift=0.0, spread=SPREAD, fresh=True)
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      set_arm(mk, g.expand(n, -1), t, 1.0)
    lift_try[n16] = 0
    go_state(n16, 1, t)
    # 持ち上がったか（箱の底がテーブルの天板から 1 cm 以上）。上がっていなければ 1 cm ずつ追加で上げる（3 回まで）。
    # 上がらないまま下がると、受け具の縁が箱を引っかけたまま進めず止まった（run14）
    # 傾いて片側だけ上がったのは「持ち上がった」としない（片方の受け具がフランジの上に乗ったまま下がると、受け具が箱の中に入って引っかかった）
    up_b = quat_apply(box.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
    lifted = ((box.data.root_link_pos_w[:, 2] - origin[:, 2]) - BIN_OUTER[2] / 2 - ha > 0.01) & (up_b[:, 2] > math.cos(math.radians(12.0)))
    if not args.lift_check:
      lifted = torch.ones_like(lifted)
    for i in torch.nonzero(m3 & ~lifted & (lift_try < 3)).flatten().tolist():
      lift_try[i] += 1
      g, _ = pick_goal(i, lift=LIFT + 0.01 * int(lift_try[i]), spread=-SQUEEZE)
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      set_arm(mk, g.expand(n, -1), t, 0.7)
      m3[i] = False
    # それでも上がらなければ、下がらずに（箱を引きずってテーブルから落とすため）手を開いてそろえ直す（2 回まで）
    rr = m3 & ~lifted & (retry_pick < args.pick_retries)
    for i in torch.nonzero(rr).flatten().tolist():
      retry_pick[i] += 1
      lift_try[i] = 0
      g, _ = pick_goal(i, lift=0.0, spread=SPREAD)
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      set_arm(mk, g.expand(n, -1), t, 1.0)
      if i == args.env_idx:
        log.append(f"t={t:.1f} 台 {i}：持ち上がらない → 手を開いてそろえ直す（{int(retry_pick[i])} 回目）")
    if rr.any():
      hold_xy[rr] = p[rr, :2]
      hold_yaw[rr] = yaw[rr]
      go_state(rr, 1, t)
      m3 = m3 & ~rr
    # 上げ直し・そろえ直しを使い切っても上がらなければ、失敗として止まる（段階 12。箱を引きずらない）
    give_up = m3 & ~lifted & (lift_try >= 3)
    go_state(give_up, 12, t)
    m3 = m3 & ~give_up
    back_from[m3] = p[m3, :2]
    t_pick[m3] = t
    go_state(m3, 4, t)
    for s_from in (2, 1):
      ms = (state == s_from) & arm_done
      if s_from == 1 and args.plate_loop:
        # 受け板の高さがそろった（4 mm 以内）か、補正を 3 回したら差し込む（腕が止まって 0.5 秒後に測ってから）
        ms = (state == 1) & (t >= t_from + dur + 0.5) & ((plate_err < 0.004) | (plate_fix >= 3) | (t - t_state > 8.0))
      for i in torch.nonzero(ms).flatten().tolist():
        mk = torch.zeros(n, dtype=torch.bool, device=device)
        mk[i] = True
        g, _ = pick_goal(i, lift=LIFT if s_from == 2 else 0.0, spread=-SQUEEZE)
        set_arm(mk, g.expand(n, -1), t, 1.5)
      go_state(ms, s_from + 1, t)
    if args.plate_loop:
      # 受け板の高さの補正（差し込む前、手を開いた構えのとき）：腕が止まって 0.5 秒たったら、左右の受け板の中心を箱の座標で測り、
      # フランジの下 PICK_CLEAR の高さとの差を、左右別々に 1 回で補正する（0.8 秒でゆっくり。最大 3 回）。
      # 押さえとフランジのすき間は 3 mm なので、差し込みの上下の許容は 3 mm しかない。
      # 0.25 秒ごとに測り直して素早く動かすと、腕の動きで方策が上体の姿勢を変え、受け板が ±3〜5 cm 揺れて収まらなかった
      m12 = (state == 1) & (t >= t_from + dur + 0.5)
      if m12.any():
        gx = env.sim.data.geom_xpos[:, hook_gid]  # 台数 × 2（右・左）× 3
        bq_i = box.data.root_link_quat_w * torch.tensor([1.0, -1, -1, -1], device=device)
        rel_p = quat_apply(bq_i[:, None, :].expand(n, 2, 4).reshape(-1, 4), (gx - box.data.root_link_pos_w[:, None, :]).float().reshape(-1, 3)).reshape(n, 2, 3)
        e = (PLATE_Z - PICK_CLEAR) - rel_p[:, :, 2]
        plate_err[m12] = e[m12].abs().max(dim=-1).values
        need = m12 & (plate_err >= 0.004) & (plate_fix < 3)
        for i in torch.nonzero(need).flatten().tolist():
          side_dz[i] = torch.clamp(side_dz[i] + torch.clamp(e[i], -0.02, 0.02), -0.04, 0.04)
          plate_fix[i] += 1
          g, _ = pick_goal(i, lift=0.0, spread=SPREAD)
          mk = torch.zeros(n, dtype=torch.bool, device=device)
          mk[i] = True
          set_arm(mk, g.expand(n, -1), t, 0.8)
          if i == args.env_idx:
            log.append(f"t={t:.1f} 台 {i}：受け板の高さの差 右 {float(e[i, 0]) * 1000:+.0f} 左 {float(e[i, 1]) * 1000:+.0f} mm → 補正（{int(plate_fix[i])} 回目）")
    elif k % 12 == 0:
      for i in torch.nonzero((state == 1) | (state == 2)).flatten().tolist():
        g, _ = pick_goal(i, lift=0.0, spread=SPREAD if int(state[i]) == 1 else -SQUEEZE)
        q_to[i] = g
    hold_base((state >= 1) & (state <= 3))
    # 4 後ろへ 0.4 m
    m4 = state == 4
    vx = torch.where(m4, torch.full_like(vx, -0.2), vx)
    d4 = torch.linalg.norm(p[:, :2] - back_from, dim=-1)
    # やり直し：2 秒たっても 5 cm も下がれない＝片方の受け具がフランジにかからず、箱がテーブルに引っかかっている。
    # 手を開いて放し（段階 13）、0.25 m 下がって（段階 14）、近づくところからやり直す（2 回まで）
    # 0.15 m 下がっても箱がテーブルの上（排斥用テーブルの天板の上、中心から 15 cm 以内・天板から 1 cm 以内）に残っている＝取り上げに失敗して空の手で下がっている（これも放してやり直す。
    # やり直さないと、空の手のまま検査用テーブルまで行って「置いた」ことになっていた）
    left_behind = (torch.linalg.norm((box.data.root_link_pos_w - origin)[:, :2] - torch.tensor([XA, 0.0], device=device), dim=-1) < 0.15) &         ((box.data.root_link_pos_w[:, 2] - origin[:, 2]) - BIN_OUTER[2] / 2 - ha < 0.01)
    st4 = m4 & (((t - t_state > 2.0) & (d4 < 0.05)) | ((d4 > 0.15) & left_behind)) & (retry_pick < args.pick_retries)
    for i in torch.nonzero(st4).flatten().tolist():
      retry_pick[i] += 1
      g, _ = pick_goal(i, lift=0.0, spread=SPREAD)
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      set_arm(mk, g.expand(n, -1), t, 1.0)
      if i == args.env_idx:
        log.append(f"t={t:.1f} 台 {i}：下がれない → 放してやり直す（{int(retry_pick[i])} 回目）")
    go_state(st4, 13, t)
    m4 = state == 4
    vx = torch.where(st4, torch.zeros_like(vx), vx)
    n13 = (state == 13) & arm_done
    back_from[n13] = p[n13, :2]
    go_state(n13, 14, t)
    m14 = state == 14
    # 放したあと 3 秒たっても 5 cm も下がれない（開いた手がまだ箱に引っかかっている）→ 手を 3 cm 上げる（1 回）
    up14 = m14 & (t - t_state > 3.0) & (torch.linalg.norm(p[:, :2] - back_from, dim=-1) < 0.05) & ~unhook14
    for i in torch.nonzero(up14).flatten().tolist():
      g, _ = pick_goal(i, lift=0.03, spread=SPREAD)
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      set_arm(mk, g.expand(n, -1), t, 0.8)
    unhook14 |= up14
    vx = torch.where(m14, torch.full_like(vx, -0.15), vx)
    n14 = m14 & (torch.linalg.norm(p[:, :2] - back_from, dim=-1) > 0.25)
    set_arm(n14, spread_q.expand(n, -1), t, 1.0)
    lift_try[n14] = 0
    unhook14[n14] = False
    go_state(n14, 0, t)
    # やり直しを使い切っても箱が残っていれば、失敗として止まる（段階 12）
    lost = m4 & (d4 > 0.15) & left_behind
    go_state(lost, 12, t)
    m4 = state == 4
    n4 = m4 & (d4 > 0.4)
    set_arm(n4, carry_q.expand(n, -1), t, 1.5)
    go_state(n4, 5, t)
    # 5 腕を抱える姿勢へ
    n5 = (state == 5) & arm_done
    go_state(n5, 6, t)
    # 6 180 度回る
    m6 = state == 6
    e6 = wrap(math.pi - yaw)
    # その場の旋回は、箱を持つと遅く（2 度/秒ほど、180 度に 45 秒）→ ゆっくり前へ歩きながら回る（半径 0.3 m ほど）
    wz = torch.where(m6, torch.sign(e6) * torch.clamp(1.5 * e6.abs(), 0.3, 0.5), wz)
    vx = torch.where(m6 & (e6.abs() > 0.15), torch.full_like(vx, 0.15), vx)
    go_state(m6 & (e6.abs() < 0.06), 7, t)
    # 7 検査用テーブルへ
    m7 = state == 7
    # 近づくときは腕で箱を上げておき、テーブルの上に来たら抱える姿勢へ戻して載せる。
    # 抱える姿勢より下へ腕を下ろすと、方策が上体を後ろへ反らせて（−3 → −14 度）釣り合いを取り、箱の高さが変わらなかった（run14）
    up7 = m7 & ~arm_up
    if up7.any():
      set_arm(up7, raise_q.expand(n, -1), t, 1.0)
      arm_up[up7] = True
    if m7.any():
      goal = torch.tensor([XB, 0.0], device=device)  # 受け具（箱）を検査用テーブルの中心の上へ
      cx, cy, cw, ok = approach(goal.expand(n, 2), torch.full((n,), math.pi, device=device), 0.04, m7, tol_y=0.05)
      vx = torch.where(m7, cx, vx)
      vy = torch.where(m7, cy, vy)
      wz = torch.where(m7, cw, wz)
      hold = torch.where(m7 & ok, hold + dt, torch.where(m7, torch.zeros_like(hold), hold))
      n7 = m7 & (hold > 1.0)
      # 下ろす量は、そのときの箱の底の高さから決める（立っているときの高さは台ごと・時間で数 cm ちがう）
      oz = BIN_OUTER[2] / 2
      for i in torch.nonzero(n7).flatten().tolist():
        bottom = float(box.data.root_link_pos_w[i, 2] - origin[i, 2]) - oz
        place_dz[i] = -PLACE_CLEAR
        place_try[i] = 0
        g, _ = arm_vec(dz=float(place_dz[i]))
        if i == args.env_idx:
          log.append(f"t={t:.1f} 台 {i}：置く前の箱の底 {bottom * 1000:.0f} mm（テーブル {hb * 1000:.0f} mm）")
        mk = torch.zeros(n, dtype=torch.bool, device=device)
        mk[i] = True
        set_arm(mk, g.expand(n, -1), t, 2.0)
      hold_xy[n7] = p[n7, :2]
      hold_yaw[n7] = yaw[n7]
      go_state(n7, 8, t)
    # 8 下ろす → 9 開く → 10 下がる（後ろから判定する）
    arm_done = t >= t_from + dur + 0.3
    n9 = (state == 9) & arm_done
    back_from[n9] = p[n9, :2]
    # 下がる前に、開いた手を 2 cm 上げる（下げて開いた手首がテーブルの天板に引っかかって下がれなかった）
    set_arm(n9, open_up_q.expand(n, -1), t, 0.6)
    go_state(n9, 10, t)
    n8 = (state == 8) & arm_done
    rest = (box.data.root_link_pos_w[:, 2] - origin[:, 2]) - BIN_OUTER[2] / 2 - hb
    for i in torch.nonzero(n8).flatten().tolist():
      mk = torch.zeros(n, dtype=torch.bool, device=device)
      mk[i] = True
      if float(rest[i]) > PLACE_REST and int(place_try[i]) < 3:
        # まだ載っていない：1 cm 追加で下げる
        place_dz[i] -= 0.005
        place_try[i] += 1
        g, _ = arm_vec(dz=float(place_dz[i]))
        set_arm(mk, g.expand(n, -1), t, 0.5)
        n8[i] = False
        continue
      if i == args.env_idx:
        log.append(f"t={t:.1f} 台 {i}：手を開く（箱の底と天板の差 {float(rest[i]) * 1000:.0f} mm、追加で下げた回数 {int(place_try[i])}）")
      g2, _ = arm_vec(dz=float(place_dz[i]), spread=SPREAD)
      set_arm(mk, g2.expand(n, -1), t, 1.0)
    go_state(n8, 9, t)
    hold_base((state == 8) | (state == 9))
    m10 = state == 10
    # やり直し：下がれないまま 4 秒たったら、手をさらに 2 cm 上げて開き直す（1 回）
    r10 = m10 & (t - t_state > 4.0) & (torch.linalg.norm(p[:, :2] - back_from, dim=-1) < 0.1) & (retry_back < 1)
    if r10.any():
      retry_back[r10] += 1
      set_arm(r10, open_up2_q.expand(n, -1), t, 0.6)
      back_from[r10] = p[r10, :2]
      t_state[r10] = t
    vx = torch.where(m10 & (t - t_state > 0.8), torch.full_like(vx, -0.2), vx)
    n10 = m10 & (torch.linalg.norm(p[:, :2] - back_from, dim=-1) > 0.3)
    set_arm(n10, carry_q.expand(n, -1), t, 1.5)
    t_done[n10] = t
    go_state(n10, 11, t)

    if k % 50 == 0:
      tq_ = robot.data.body_link_quat_w[:, torso_id]
      fwd_ = quat_apply(tq_, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
      torso_pitch = torch.asin(torch.clamp(-fwd_[:, 2], -1, 1))  # 上体の前傾（前に傾くと正）
      if args.debug_contacts:
        i = args.env_idx
        log.append(f"t={t:.0f} 腕の目標 {[round(v, 3) for v in act.arm_target[i].tolist()]}、実際 {[round(v, 3) for v in robot.data.joint_pos[i, robot.find_joints(tuple(names), preserve_order=True)[0]].tolist()]}")
      i = args.env_idx
      log.append(f"t={t:.0f} 段階 {int(state[i])} 胴体 x {p[i, 0]:+.3f} y {p[i, 1]:+.3f} z {p[i, 2]:.3f} 上体の前傾 {math.degrees(float(torso_pitch[i])):+.1f} 度 向き {math.degrees(yaw[i]):+.0f} 度、指令 vx {vx[i]:+.2f} vy {vy[i]:+.2f} wz {wz[i]:+.2f}、箱 x {float(box.data.root_link_pos_w[i, 0] - origin[i, 0]):+.3f} z {float(box.data.root_link_pos_w[i, 2]):.3f}"
                 f"、箱の傾き {math.degrees(math.acos(max(-1.0, min(1.0, float(quat_apply(box.data.root_link_quat_w[i:i + 1], torch.tensor([[0.0, 0.0, 1.0]], device=device))[0, 2]))))):.1f} 度"
                 f"、受け具の箱の中心 x {pocket[i, 0]:+.3f} y {pocket[i, 1]:+.3f} z {float(mid[i, 2] + off_mid[2]):.3f}")
    if args.debug_contacts and k % 10 == 0 and 1 <= int(state[args.env_idx]) <= 4:
      if not hasattr(env, "_fg_hook_gid"):
        env._fg_hook_gid = [env.sim.mj_model.geom(f"robot/{sd}_hook_plate_collision").id for sd in ("right", "left")]
      gx = env.sim.data.geom_xpos[args.env_idx, env._fg_hook_gid]  # 世界
      bq = box.data.root_link_quat_w[args.env_idx:args.env_idx + 1]
      rel = quat_apply(bq * torch.tensor([1.0, -1, -1, -1], device=device), (gx - box.data.root_link_pos_w[args.env_idx]).float())
      log.append(f"t={t:.1f} 段階 {int(state[args.env_idx])} 箱 z {float(box.data.root_link_pos_w[args.env_idx, 2]):.3f} 受け板（箱の座標）右 x {rel[0, 0]:+.3f} y {rel[0, 1]:+.3f} z {rel[0, 2]:+.3f}、左 x {rel[1, 0]:+.3f} y {rel[1, 1]:+.3f} z {rel[1, 2]:+.3f}、胴体の向き {math.degrees(yaw[args.env_idx]):+.1f} 度、補正 右 {float(side_dz[args.env_idx, 0]) * 1000:+.0f} 左 {float(side_dz[args.env_idx, 1]) * 1000:+.0f} mm")
    if args.debug_contacts and k % 10 == 0:
      wd = env.sim.wp_data
      mm = env.sim.mj_model
      na = int(wd.nacon.numpy()[0])
      gg = wd.contact.geom.numpy()[:na]
      ww = wd.contact.worldid.numpy()[:na]
      cs = set()
      for (g1, g2), w in zip(gg, ww):
        if w == args.env_idx:
          n1, n2 = mm.geom(int(g1)).name, mm.geom(int(g2)).name
          if ("bin" in n1 + n2 or "table" in n1 + n2) and not ("bin" in n1 + n2 and "table" in n1 + n2) and "terrain" not in n1 + n2:
            cs.add(f"{n1}—{n2}")
      if cs:
        log.append(f"t={t:.1f} 接触 " + "、".join(sorted(cs)))
    # 取り上げ（そろえる〜持ち上げる）の間は、腰の yaw・roll を 0 に固定して、上体の向きを胴体にそろえる
    act.waist_target = torch.zeros(n, 3, device=device)
    pick_m = (state >= 1) & (state <= 3)
    if not args.lock_waist_pick:
      pick_m = pick_m & False
    place_m = ((state == 8) | (state == 9)) & args.lock_waist
    act.waist_mask = torch.stack([pick_m, pick_m, place_m], dim=-1)
    if t < 1.2:
      vx, vy, wz = vx * 0, vy * 0, wz * 0  # 最初は腕を構えるだけ（構える前に歩くと、受け具が箱に当たる）
    live = ~fell & ~dropped  # 箱を落としたら止まる（床の箱を追いかけない）
    cmd.vel_command_b[:, 0] = torch.where(live, vx, 0.0)
    cmd.vel_command_b[:, 1] = torch.where(live, vy, 0.0)
    cmd.vel_command_b[:, 2] = torch.where(live, wz, 0.0)

    with torch.no_grad():
      obs, _, dones, _ = wenv.step(policy(obs))
    newf = dones.bool() & ~fell
    if newf.any():
      for i in torch.nonzero(newf).flatten().tolist():
        fall_info.append((i, round(t, 1), int(state[i]), bool(dropped[i])))
    fell |= dones.bool()
    bz = box.data.root_link_pos_w[:, 2]
    newd = (bz < 0.5) & ~fell & ~dropped
    for i in torch.nonzero(newd).flatten().tolist():
      drop_info.append(int(state[i]))
    dropped |= (bz < 0.5) & ~fell
    if args.video:
      frames.append(env.render())

  # 結果：検査用テーブルの上に、まっすぐ置けたか
  bp = box.data.root_link_pos_w - origin
  up = quat_apply(box.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
  tilt = torch.rad2deg(torch.acos(torch.clamp(up[:, 2], -1, 1)))
  oz = BIN_OUTER[2] / 2
  on_b = ((bp[:, 0] - XB).abs() < 0.10) & (bp[:, 1].abs() < 0.08) & ((bp[:, 2] - (hb + oz)).abs() < 0.01) & (tilt < 5)
  ok = on_b & ~fell & ~dropped & (state == 11)
  print(f"checkpoint: {args.checkpoint}")
  print(f"テーブルの高さ：排斥用 {ha * 1000:.0f} mm、検査用 {hb * 1000:.0f} mm（抱えた箱の底 {zc * 1000:.0f} mm）、テーブルの間 {XA - XB:.1f} m")
  print(f"成功率: {ok.float().mean().item() * 100:.1f} %（成功 {int(ok.sum())} / {n} 台、転倒 {int(fell.sum())}、箱を落とした {int(dropped.sum())}、"
        f"置いたが位置・傾きが外れ {int((~on_b & (state == 11) & ~fell & ~dropped).sum())}、時間内に終わらず {int(((state < 11) & ~fell).sum())}）")
  for s in range(17):
    c = int(((state == s) & ~fell).sum())
    if c and s != 11:
      print(f"  止まった段階 {s}: {c} 台")
  if ok.any():
    print(f"  取り上げまで 平均 {torch.nanmean(t_pick).item():.1f} 秒、完了まで 平均 {torch.nanmean(t_done).item():.1f} 秒")
  print(f"  置いた箱：位置のずれ 平均 x {(bp[ok, 0] - XB).abs().mean().item() * 100 if ok.any() else float('nan'):.1f} cm、"
        f"傾き 平均 {tilt[ok].mean().item() if ok.any() else float('nan'):.1f} 度")
  off = (~on_b & (state == 11) & ~fell & ~dropped)
  if off.any():
    print("  置いたが外れた台（x のずれ, y, 天板からの高さ mm, 傾き 度）: " + " ".join(
        f"({(bp[i, 0] - XB) * 100:+.1f},{bp[i, 1] * 100:+.1f},{(bp[i, 2] - hb - oz) * 1000:+.0f},{tilt[i]:.0f})" for i in torch.nonzero(off).flatten().tolist()))
  print("  台ごとの最後の段階: " + " ".join(f"{i}:{int(state[i])}" for i in range(n)))
  if drop_info:
    from collections import Counter
    print("  箱を落としたときの段階（段階: 台数）: " + " ".join(f"{a}:{b}" for a, b in sorted(Counter(drop_info).items())))
  if fall_info:
    print("  転倒（台, 時刻, 段階, 箱を落とした後か）: " + " ".join(f"({a},{b},{c},{'落' if d else '-'})" for a, b, c, d in fall_info))
  for line in log:
    print("  " + line)
  if args.debug_contacts:
    wd = env.sim.wp_data
    m = env.sim.mj_model
    na = int(wd.nacon.numpy()[0])
    gg = wd.contact.geom.numpy()[:na]
    ww = wd.contact.worldid.numpy()[:na]
    seen = set()
    for (g1, g2), w in zip(gg, ww):
      if w != args.env_idx:
        continue
      n1, n2 = m.geom(int(g1)).name, m.geom(int(g2)).name
      if any(x in n1 + n2 for x in ("table", "bin", "can", "hook")) and (n1, n2) not in seen:
        seen.add((n1, n2))
        print(f"  接触: {n1} — {n2}")
  if args.video and frames:
    import imageio.v3 as iio

    iio.imwrite(args.video, np.stack(frames), fps=int(round(1 / dt)))
    print(f"動画: {args.video}")


if __name__ == "__main__":
  main()
