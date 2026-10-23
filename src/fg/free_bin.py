"""落ちる回収箱（手首に固定しない）のモデル（mjlab 1.6.0）。

- 箱は別の物体（自由に動く、mjlab の Entity "bin"）。フランジが手の甲に乗る接触と摩擦だけで支える
- 当たり判定：箱の geom は contype=2・conaffinity=1。ロボットの手首 yaw のリンク（手）だけ conaffinity を 1|2 にする
  → 箱は「床・段・テーブル」と「手」にだけ当たる（胴・腕の他の部分とは当たらない：簡略化）
- 缶は箱に固定（中で転がらない：簡略化）
- リセット：ロボットのリセットのあと、胴体の位置・向きから「抱える姿勢での箱の位置」に置く（2 mm 浮かせて落とす）
- 落とした判定：箱が、左手首から見た決まった位置から一定以上（既定 8 cm）離れた → 終了
"""

from dataclasses import replace

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.spec_config import CollisionCfg

from fg.luna_constants import (
  CARRY_ARM_POSE,
  FOOT_GEOM_EXPR,
  KNEES_BENT_KEYFRAME,
  add_bin_geoms,
  bin_layout,
  get_spec,
)

HOOK_GEOM_EXPR = r"^(left|right)_hook_.*_collision$"
# 受け具（手首 yaw のリンクに付ける、フランジを受けるコの字の溝）の寸法 [m]
HOOK_PLATE_T = 0.012  # 受け板の厚さ（4 mm では、歩き出しで箱が受け具をすり抜けて外れることがあった。薄い板どうしの接触）
HOOK_LIP_H = 0.025  # 抜け止めの縁の高さ（受け板の上面から）。12 mm では 20 cm の下り 0.78 m/s で箱が跳ねて 86 % 落ちた
HOOK_LIP_T = 0.012  # 縁の厚さ
HOOK_CLEAR = 0.002  # フランジと外側の縁のすき間
HOOK_CLEAR_X = 0.015  # フランジと前後の縁のすき間。横から差し込むとき、前後の位置合わせの許容（±15 mm）になる
HOOK_INNER = -0.002  # 受け板の内側の端と胴の壁の外側のすき間（負＝壁より外）。横から差し込むとき壁に当たらないように 2 mm 空ける
HOOK_DENSITY = 2700.0  # アルミ
HOOK_CHAMFER = float(__import__("os").environ.get("FG_HOOK_CHAMFER", "0.006"))  # 受け板の内側の上の角の面取り（45 度）[m]。
# 押さえのすき間 3 mm では、横から差し込むときの上下の許容が 3 mm しかなく、取り上げの失敗の主因だった。押さえの斜面は「受け具が低すぎる」を、
# この面取りは「受け具が高すぎる」を受け止める（2026-10-12 15:30。実験用に環境変数 FG_HOOK_CHAMFER、0 で面取りなし）
# 抜け止めの押さえ（フランジの上にかぶさる板）と、差し込むときの案内の斜面（計画書 7.3 の「抜け止め」「位置ずれを吸収するガイド」）。
# 押さえが無いと、歩く・段を下りるときに箱が受け板の上で傾いて外れた（落とす直前の傾き 25〜34 度、2026-10-12 朝）
HOOK_KEEP_GAP = float(__import__("os").environ.get("FG_KEEP_GAP", "0.003"))  # 押さえの下面とフランジの上面のすき間（抱えたとき。実験用に環境変数 FG_KEEP_GAP）。
# 取り上げで差し込みやすくするため 15 mm に広げたら、20 cm の上り 0.58 m/s で 3 mm 100 % → 8 mm 90.2 % → 15 mm 86.7 % に落ちた（run14、256 台）
# → 3 mm のまま。差し込みの上下の精度は、受け板の位置を測って合わせる閉ループで出す（pick_place.py）
HOOK_KEEP_W = 0.010  # 押さえがフランジにかぶさる幅（外側から）
HOOK_KEEP_T = 0.012  # 押さえの厚さ
HOOK_WEDGE = 0.015  # 押さえの内側の斜面（45 度）の水平の長さ。差し込むとき、高すぎるフランジを押さえの下へ案内する
BIN_HAND_FRICTION = (0.8, 0.005, 0.0001)  # 手の甲とフランジの摩擦（プラスチックとゴムの目安【仮】）
DROP_DISTANCE = 0.04  # [m] フランジの帯が受け具の決まった位置から 4 cm 以上ずれたら「外れた」
# 箱を持つときの腕の PD の硬さの倍率。歩行の学習の値（固有振動数 10 Hz から決めた、肩 kp 約 70 Nm/rad）のままだと、
# 手首の横の硬さが約 350 N/m しかなく、歩くと両手が 3〜4 cm 開いて箱が間から落ちた（2026-10-12 朝、手首の間隔 36.6 → 40 cm）
ARM_KP_SCALE = 8.0
ARM_KD_SCALE = 2.83  # 減衰比を保つ（kd ∝ √kp）
SPAWN_LIFT = 0.002  # [m]

# ロボットの当たり判定：今まで（地面とだけ）＋受け具は箱とも当たる（手のメッシュは箱と当たらない）
COLLISION_HANDS_TOUCH_BIN = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  contype=0,
  conaffinity={HOOK_GEOM_EXPR: 3, ".*_collision.*": 1},
  condim={FOOT_GEOM_EXPR: 3, HOOK_GEOM_EXPR: 3, ".*_collision.*": 1},
  priority={FOOT_GEOM_EXPR: 1, ".*": 0},
  friction={FOOT_GEOM_EXPR: (0.6,), HOOK_GEOM_EXPR: BIN_HAND_FRICTION},
)

_LAYOUT = None


def layout() -> dict:
  global _LAYOUT
  if _LAYOUT is None:
    _LAYOUT = bin_layout()
  return _LAYOUT


def add_hooks(spec: mujoco.MjSpec) -> None:
  """左右の手首 yaw のリンクに、フランジを受けるコの字の受け具（ロボット適合型のフック）を付ける。

  形：フランジ（箱の左右の、胴から外へ張り出した帯）を横から挟むコの字。下に受け板、外側と前後の端に縁、
  上にフランジの外側 10 mm にかぶさる押さえ（抜け止め）と、その内側に 45 度の案内の斜面。
  → フランジは溝の中で、上・横に動けない（前後は ±15 mm まで）。テーブルの上の箱に、横から差し込んで持ち、置いたら横へ抜く
  取り上げ：手を外へ開いてフランジの高さに構え、内へ寄せて差し込み、持ち上げる
  寸法は抱える姿勢での箱の位置（bin_layout）から決める。受け板の上面＝フランジの下面
  """
  import numpy as np

  from fg.luna_constants import BIN_FLANGE_T, BIN_INNER, BIN_OUTER, BIN_WALL

  L = layout()
  d, center = L["data"], L["center_w"]
  ox, oy, oz = (v / 2 for v in BIN_OUTER)
  by = BIN_INNER[1] / 2 + BIN_WALL
  fz = oz - BIN_FLANGE_T  # フランジの下面（箱の座標）
  xl = ox + HOOK_CLEAR_X + HOOK_LIP_T  # 受け板の前後の端
  yin, yout = by - HOOK_INNER, oy + HOOK_CLEAR + HOOK_LIP_T
  for side, sgn, bid in (("left", 1.0, L["lw"]), ("right", -1.0, L["rw"])):
    # 箱の座標での部品（中心、半分の大きさ）。左は +y、右は −y
    yc = sgn * (yin + yout) / 2
    # 受け板の内側の上の角を 45 度に面取り（HOOK_CHAMFER）：差し込むとき受け具が高すぎても、フランジの縁が斜面に乗って受け板の上へ案内される
    ch = HOOK_CHAMFER
    yp_in = yin + ch
    parts = {
      "plate": ((0.0, sgn * (yp_in + yout) / 2, fz - HOOK_PLATE_T / 2), (xl, (yout - yp_in) / 2, HOOK_PLATE_T / 2)),
      "lip_out": ((0.0, sgn * (yout - HOOK_LIP_T / 2), fz + HOOK_LIP_H / 2), (xl, HOOK_LIP_T / 2, HOOK_LIP_H / 2)),
      "lip_front": ((xl - HOOK_LIP_T / 2, yc, fz + HOOK_LIP_H / 2), (HOOK_LIP_T / 2, (yout - yin) / 2, HOOK_LIP_H / 2)),
      "lip_back": ((-xl + HOOK_LIP_T / 2, yc, fz + HOOK_LIP_H / 2), (HOOK_LIP_T / 2, (yout - yin) / 2, HOOK_LIP_H / 2)),
    }
    # 押さえ：フランジの外側 HOOK_KEEP_W の上に、HOOK_KEEP_GAP 空けて
    kz = oz + HOOK_KEEP_GAP + HOOK_KEEP_T / 2
    ky_in = oy - HOOK_KEEP_W
    parts["keeper"] = ((0.0, sgn * (ky_in + yout) / 2, kz), (xl, (yout - ky_in) / 2, HOOK_KEEP_T / 2))
    # 斜面：押さえの内側の端から、内側・上へ 45 度
    wl = HOOK_WEDGE * 2**0.5
    wc_y = ky_in - HOOK_WEDGE / 2
    wc_z = oz + HOOK_KEEP_GAP + HOOK_WEDGE / 2
    wedge = ((0.0, sgn * wc_y, wc_z), (xl, wl / 2, HOOK_KEEP_T / 2), -sgn * 45.0)
    xw, Rw = d.xpos[bid], d.xmat[bid].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rw.T.flatten())  # 箱（世界の向き）→ 手首の座標
    body = spec.body(f"{side}_wrist_yaw_link")
    items = [(nm, c, h, 0.0) for nm, (c, h) in parts.items()] + [("wedge", *wedge)]
    if ch > 0:
      # 面取りの斜面：上面が (yin, fz − ch) から (yin + ch, fz) へ上がる 45 度の板（厚さは受け板と同じ）
      rl = ch * 2**0.5
      off = HOOK_PLATE_T / 2 / 2**0.5
      items.append(("ramp", (0.0, sgn * (yin + ch / 2 + off), fz - ch / 2 - off), (xl, rl / 2, HOOK_PLATE_T / 2), sgn * 45.0))
    for nm, c, h, ang in items:
      pos = Rw.T @ (center + np.array(c) - xw)
      # 箱の座標での向き（x 軸まわりに ang 度）→ 手首の座標
      a = np.radians(ang)
      qx = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
      qg = np.zeros(4)
      mujoco.mju_mulQuat(qg, q, qx)
      for kind in ("collision", "visual"):
        body.add_geom(name=f"{side}_hook_{nm}_{kind}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), pos=list(pos),
                      quat=list(qg), rgba=[0.75, 0.75, 0.78, 1.0],
                      density=HOOK_DENSITY if kind == "collision" else 0.0,
                      group=3 if kind == "collision" else 2,
                      contype=0, conaffinity=3 if kind == "collision" else 0)


def get_spec_with_hooks() -> mujoco.MjSpec:
  spec = get_spec()
  add_hooks(spec)
  return spec


def get_free_bin_spec(num_cans: int):
  def fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="reject_bin")
    b.add_freejoint(name="reject_bin_joint")
    add_bin_geoms(b, num_cans, contype=2, conaffinity=1, condim=3, friction=BIN_HAND_FRICTION)
    return spec

  return fn


def reset_bin_on_hands(env, env_ids, bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
                       robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
  """ロボットのリセットのあとに呼ぶ：胴体の位置・向き（qpos から直接読む）から箱を手の上に置く。"""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot = env.scene[robot_cfg.name]
  box = env.scene[bin_cfg.name]
  qa = robot.indexing.free_joint_q_adr
  va = robot.indexing.free_joint_v_adr
  qpos = env.sim.data.qpos[env_ids][:, qa]
  qvel = env.sim.data.qvel[env_ids][:, va]
  pos, quat = qpos[:, 0:3], qpos[:, 3:7]
  off = torch.tensor(layout()["center_root"], dtype=torch.float32, device=env.device)
  off = off + torch.tensor([0.0, 0.0, SPAWN_LIFT], device=env.device)
  p = pos + quat_apply(quat, off.expand(len(env_ids), 3))
  pose = torch.cat([p, quat], dim=-1)
  box.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  vel = torch.zeros(len(env_ids), 6, device=env.device)
  vel[:, 0:3] = qvel[:, 0:3]  # 自由関節の並進速度は世界座標
  box.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def _flange_points():
  """左右のフランジの帯の中心（箱の座標）と、抱える姿勢でそれが来る位置（各手首の座標）。"""
  import numpy as np

  from fg.luna_constants import BIN_FLANGE_T, BIN_INNER, BIN_OUTER, BIN_WALL

  L = layout()
  d, center = L["data"], L["center_w"]
  ox, oy, oz = (v / 2 for v in BIN_OUTER)
  by = BIN_INNER[1] / 2 + BIN_WALL
  out = []
  for sgn, bid in ((1.0, L["lw"]), (-1.0, L["rw"])):
    c_b = np.array([0.0, sgn * (by + oy) / 2, oz - BIN_FLANGE_T / 2])
    Rw = d.xmat[bid].reshape(3, 3)
    out.append((c_b, Rw.T @ (center + c_b - d.xpos[bid])))
  return out


def bin_offset_from_hand(env, bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
                         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """左右のフランジの帯が、それぞれの受け具の決まった位置からどれだけずれたか（大きいほう）[m]。

  （最初は「左手首から見た箱の中心のずれ」で判定したが、箱が両方の受け具に入ったまま手ごと傾くと、
  箱の中心が左手首から 18 cm 離れているぶん大きく出て、落としていないのに落としたと判定していた）
  """
  robot = env.scene[robot_cfg.name]
  box = env.scene[bin_cfg.name]
  if not hasattr(env, "_fg_flange"):
    ids, _ = robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"))
    env._fg_wrist_ids = ids
    env._fg_flange = [(torch.tensor(cb, dtype=torch.float32, device=env.device),
                       torch.tensor(cw, dtype=torch.float32, device=env.device)) for cb, cw in _flange_points()]
  if not hasattr(env, "_fg_lw_id"):
    env._fg_lw_id = env._fg_wrist_ids[0]
  bp, bq = box.data.root_link_pos_w, box.data.root_link_quat_w
  worst = torch.zeros(env.num_envs, device=env.device)
  for (cb, cw), wid in zip(env._fg_flange, env._fg_wrist_ids):
    wp = robot.data.body_link_pos_w[:, wid]
    wq = robot.data.body_link_quat_w[:, wid]
    actual = bp + quat_apply(bq, cb.expand(env.num_envs, 3))
    target = wp + quat_apply(wq, cw.expand(env.num_envs, 3))
    worst = torch.maximum(worst, torch.linalg.norm(actual - target, dim=-1))
  return worst


def bin_diag(env) -> torch.Tensor:
  """失敗の分析用：(決まった位置からのずれ、そのうち上下の成分（+ は上）、箱の傾き [度]、手首の高さ − 胴体の高さ、左右の手首の間隔)。"""
  robot = env.scene["robot"]
  box = env.scene["bin"]
  bin_offset_from_hand(env)
  wp = robot.data.body_link_pos_w[:, env._fg_lw_id]
  wq = robot.data.body_link_quat_w[:, env._fg_lw_id]
  pos_l = torch.tensor(layout()["pos_l"], dtype=torch.float32, device=env.device).expand(env.num_envs, 3)
  diff = box.data.root_link_pos_w - (wp + quat_apply(wq, pos_l))
  up = quat_apply(box.data.root_link_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
  tilt = torch.rad2deg(torch.acos(torch.clamp(up[:, 2], -1.0, 1.0)))
  if not hasattr(env, "_fg_rw_id"):
    ids, _ = robot.find_bodies(("right_wrist_yaw_link",))
    env._fg_rw_id = ids[0]
  span = torch.linalg.norm(wp - robot.data.body_link_pos_w[:, env._fg_rw_id], dim=-1)
  return torch.stack([torch.linalg.norm(diff, dim=-1), diff[:, 2], tilt, wp[:, 2] - robot.data.root_link_pos_w[:, 2], span], dim=-1)


def bin_dropped(env, distance: float = DROP_DISTANCE) -> torch.Tensor:
  return bin_offset_from_hand(env) > distance


def apply_free_bin(env_cfg, cans: int) -> None:
  """環境の設定を「落ちる箱を両手の甲に乗せた状態」にする（腕は抱える姿勢）。"""
  robot = env_cfg.scene.entities["robot"]
  arm_keys = ("shoulder", "elbow")
  joint_pos = {k: v for k, v in KNEES_BENT_KEYFRAME.joint_pos.items() if not any(a in k for a in arm_keys)}
  joint_pos.update(CARRY_ARM_POSE)
  from fg.luna_constants import LUNA_ACTUATOR_ELBOW, LUNA_ACTUATOR_SHOULDER, LUNA_ACTUATOR_SMALL

  stiff = {id(a) for a in (LUNA_ACTUATOR_SHOULDER, LUNA_ACTUATOR_ELBOW, LUNA_ACTUATOR_SMALL)}
  acts = tuple(
    replace(a, stiffness=a.stiffness * ARM_KP_SCALE, damping=a.damping * ARM_KD_SCALE) if id(a) in stiff else a
    for a in robot.articulation.actuators
  )
  env_cfg.scene.entities["robot"] = replace(
    robot,
    articulation=replace(robot.articulation, actuators=acts),
    spec_fn=get_spec_with_hooks,
    collisions=(COLLISION_HANDS_TOUCH_BIN,),
    init_state=replace(robot.init_state, joint_pos=joint_pos),
  )
  # 初期位置は高く（モデルを作るときに床とめり込まないように。リセットで手の上に置き直す）
  env_cfg.scene.entities["bin"] = EntityCfg(spec_fn=get_free_bin_spec(cans), init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)))
  # 箱の接触の分だけ、接触と拘束の上限を増やす（1 台あたり）
  if env_cfg.sim.nconmax is not None:
    env_cfg.sim.nconmax += 40
  env_cfg.sim.njmax = max(env_cfg.sim.njmax or 0, 300) + 200
  # ロボットのリセット（reset_base・reset_robot_joints）のあとに置く。dict の順に呼ばれる
  env_cfg.events["reset_bin"] = EventTermCfg(func=reset_bin_on_hands, mode="reset", params={})
  env_cfg.terminations["bin_dropped"] = TerminationTermCfg(func=bin_dropped, params={})
