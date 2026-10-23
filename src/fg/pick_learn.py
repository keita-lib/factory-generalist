"""V16：取り上げ・置く動作も方策に学習させる（腕も方策が動かす。2026-10-12 夜、作者の決定）。

シーケンス制御（pick_place.py：状態機械＋逆運動学＋歩行の方策）は 192 台で 38 % が上限だった。
主な原因は、腕を外から動かすと方策が上体の姿勢を変えて打ち消すこと（方策が腕の動きを知らない）。
→ 腕の行動も方策に任せ、取り上げ・置く動作に報酬を付けて学習させる（強化学習だけ。模倣学習は使わない）。

台ごとに 3 つの課題のどれかを割り当てる（リセットのたび）
  0 歩く（V14 と同じ：速度の指令に従う。腕は外から動かす）
  1 取り上げ：排斥用テーブルの箱へ寄り、受け具を差し込み、持ち上げ、テーブルから離れる
  2 置く：箱を抱えて検査用テーブルへ寄り、載せ、手を抜き、離れる
取り上げ・置くは、途中の場面から始める（テーブルの目の前で受け具を構えた所、少し離れた所、遠い所）。
場面はシーケンス制御と同じ計算（抱える姿勢の箱の位置・逆運動学の腕の姿勢）で作る。
差し込みの上下の許容が 3 mm しかないので、最初から通しでやらせると、偶然の成功がほぼ起きないため

入力：run14 の 280 個（歩く課題の指令は、取り上げ・置くでは 0）に、課題の 22 個を足す（task_obs）。
シミュレーターの正確な箱・テーブルの位置を使う（Phase 1 の特権情報）
重み：run14 から引き継ぐ（足した入力の重みは 0、腕の出力は 0 から：src/fg/dev/expand_checkpoint.py）
"""

import math
import os
from dataclasses import dataclass, fields

import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from fg.carry_steps import ArmScheduleAction, bin_dropped_carrying, is_empty, luna_carry_steps_v14_env_cfg
from fg.free_bin import HOOK_CHAMFER, layout
from fg.luna_constants import BIN_OUTER, KNEES_BENT_KEYFRAME
from fg.pick_place import PLATE_Y, PLATE_Z, SPREAD, TOP_X, TOP_Y, table_spec
from fg.velocity_steps import luna_steps_ppo_runner_cfg

PICK_TASK_ID = "FG-Pick-Place-Luna-V16"

# 課題の割合（歩く・取り上げ・置く）
P_MODE = tuple(float(v) for v in os.environ.get("FG_P_MODE", "0.3,0.4,0.3").split(","))
# テーブル（各台の区画の座標）。ロボットは +x を向いて、テーブルの手前に立つ
# 段差の地形で学習する（FG_V16_TERRAIN=steps）：20 cm × 2 段のステップの区画。区画の原点は台の中心・台の上面。
# テーブルは台の上（x = 0.55 m）に置く（統合と同じ配置）。歩く台は台の上から始まり、段を降り・上る（段差の力を保つ）。
# 平らな床だけで学習した run16f〜i は段差の力を失い（20 cm 上り 0 %）、統合で台の上の取り上げに転んだ
V16_STEPS = os.environ.get("FG_V16_TERRAIN", "") == "steps"
TX = 0.55 if V16_STEPS else 1.0
ZC = 0.946  # 設計どおりのロボットが抱えた箱の底の高さ [m]（measure_zc.py、run14）
HA = ZC  # 排斥用テーブルの高さ（pick_place の最終の設定：抱えた箱の底 ＋ 1 cm − 1 cm）
HB = ZC - 0.008  # 検査用テーブル（抱えた箱の底より 8 mm 低い）
TABLE_DZ = 0.010  # テーブルの高さの乱数 ±[m]（乱数ありのロボットは抱えた箱の高さが台ごとに ±1.4 cm ちがう）
OZ = BIN_OUTER[2] / 2
N_TASK_OBS = 22
ARM_ACT_SCALE = 0.25  # [rad] 取り上げ・置くの腕の行動 1 あたり
ARM_ACT_CLIP = 3.5  # 腕の行動の上限（±0.875 rad。構え→抱える姿勢は最大 0.82 rad（手首 yaw）なので、構えと抱える姿勢の間は届く）
ARM_MAX_VEL = 1.5  # [rad/s] 腕の目標の変化の上限
ARM_FILTER = 0.2  # 取り上げ・置くの腕の目標のなめらかさ（1 ステップ 20 ms ごとに差の 2 割を追う：時定数 約 0.1 秒）
PLACE_SEATED_START = float(os.environ.get("FG_PLACE_SEATED", "0.3"))  # 置く課題のうち「載せた後」から始める割合
PLACE_RAISE = 0.03  # 置く課題の始め：腕で箱を抱える姿勢より 3 cm 上げておく（pick_place の --place-raise と同じ）


def task_mode(env) -> torch.Tensor:
  """台ごとの課題（0 歩く・1 取り上げ・2 置く）。"""
  if not hasattr(env, "_fg_mode"):
    env._fg_mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._fg_table_top = torch.full((env.num_envs,), HA, device=env.device)
    env._fg_seated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env._fg_mode


def _yaw(q: torch.Tensor) -> torch.Tensor:
  return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  z = torch.zeros_like(yaw)
  return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], dim=-1)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
  return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _plate_ids(env):
  if not hasattr(env, "_fg_plate_gid"):
    m = env.sim.mj_model
    env._fg_plate_gid = torch.tensor([m.geom(f"robot/{s}_hook_plate_collision").id for s in ("left", "right")],
                                     device=env.device, dtype=torch.long)
    # 受け板の中心：面取り（HOOK_CHAMFER）で受け板の内側の端を削った分、PLATE_Y より外へずれる（2026-10-13）
    py = PLATE_Y + HOOK_CHAMFER / 2
    env._fg_plate_goal = torch.tensor([[0.0, py, PLATE_Z], [0.0, -py, PLATE_Z]], device=env.device)
  return env._fg_plate_gid, env._fg_plate_goal


def task_state(env) -> dict:
  """取り上げ・置くの判定に使う量（区画の座標）。"""
  mode = task_mode(env)
  robot, box = env.scene["robot"], env.scene["bin"]
  org = env.scene.env_origins
  bp = box.data.root_link_pos_w - org
  bq = box.data.root_link_quat_w
  rp = robot.data.root_link_pos_w - org
  rq = robot.data.root_link_quat_w
  gid, goal = _plate_ids(env)
  gx = env.sim.data.geom_xpos[:, gid] - org[:, None, :]  # 台数 × 2 × 3
  n = env.num_envs
  rel = quat_apply(_quat_inv(bq)[:, None, :].expand(n, 2, 4).reshape(-1, 4), (gx - bp[:, None, :]).float().reshape(-1, 3)).reshape(n, 2, 3)
  hook_err = rel - goal[None]  # 受け板の中心の、フランジの下の決まった位置からのずれ（箱の座標）
  up = quat_apply(bq, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
  tilt = torch.acos(torch.clamp(up[:, 2], -1.0, 1.0))
  top = env._fg_table_top
  gap = bp[:, 2] - OZ - top  # 箱の底と天板の差
  # テーブルの中心（区画の座標）。統合（integrate.py）では台ごと・段階ごとに env._fg_table_xy で変える
  table_xy = env._fg_table_xy if getattr(env, "_fg_table_xy", None) is not None else torch.tensor([TX, 0.0], device=env.device).expand(n, 2)
  dxy = torch.linalg.norm(bp[:, :2] - table_xy, dim=-1)  # 箱とテーブルの中心の水平の距離
  rup = quat_apply(rq, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3))
  # 立っている：胴体の原点が 0.72 m 以上・胴体の傾き 20 度以内（転びながらの「成功」を数えないため。run16 の最初の試行で、
  # 置く課題で後ろへ転ぶと箱が天板に落ち、胴体が箱から離れて「成功」になり、転び方を覚えた）
  # 胴体の高さはテーブルの天板から測る（統合では床が台より 0.4 m 低いので、原点からの高さでは床の上で「立っていない」になった）
  standing = (rp[:, 2] - env._fg_table_top > -0.22) & (rup[:, 2] > math.cos(math.radians(20.0)))
  bv = box.data.root_link_lin_vel_w
  # 置く：受け具に掛けたまま（受け板のずれ 3 cm 以内）、箱が天板にゆっくり（0.05 m/s 未満）載ったことがあるか（投げ落として「載った」を防ぐ）
  if not hasattr(env, "_fg_seated"):
    env._fg_seated = torch.zeros(n, dtype=torch.bool, device=env.device)
  return {"mode": mode, "bp": bp, "bq": bq, "rp": rp, "rq": rq, "hook_err": hook_err, "standing": standing, "bv": bv,
          "hook_dist": torch.linalg.norm(hook_err, dim=-1).mean(dim=-1), "tilt": tilt, "gap": gap, "dxy": dxy,
          "table_xy": table_xy, "top": top}


def _cached(env) -> dict:
  k = int(env.common_step_counter)
  if getattr(env, "_fg_ts_step", None) != k:
    env._fg_ts = task_state(env)
    env._fg_ts_step = k
  return env._fg_ts


# ---------------------------------------------------------------- 入力


def task_obs(env) -> torch.Tensor:
  """課題の入力 22 個：課題の種類（3）、目標の位置・向き（5）、箱の位置・向き・傾き（6）、受け板のずれ（6、×10）、箱の底と天板の差（×20）・天板の高さ（2）。
  位置は胴体の原点から、胴体の向き（yaw）で見た値。歩く課題では課題の種類以外は 0"""
  s = task_state(env)
  mode = s["mode"]
  n = env.num_envs
  ryaw = _yaw(s["rq"])
  qi = _quat_inv(_yaw_quat(ryaw))

  def rel(p):
    return quat_apply(qi, p - s["rp"])

  top = s["top"]
  table_c = torch.cat([s["table_xy"], (top + OZ)[:, None]], dim=-1)
  tgt = torch.where((mode == 1)[:, None], s["bp"], table_c)
  # 置くの目標の向き＝テーブルの向き（学習はテーブルが +x 側で 0。統合の製品テーブルは −x 側でロボットが −x を向くので π を
  # env._fg_table_yaw で渡す。以前は世界の 0 に固定で、統合では cos が −1 になり、学習した置くが 74 % → 6 % に崩れた：2026-10-13 15:40）
  table_yaw = env._fg_table_yaw if getattr(env, "_fg_table_yaw", None) is not None else torch.zeros_like(ryaw)
  tgt_yaw = torch.where(mode == 1, _yaw(s["bq"]), table_yaw) - ryaw
  byaw = _yaw(s["bq"]) - ryaw
  out = torch.cat([
    torch.nn.functional.one_hot(mode, 3).float(),
    rel(tgt), torch.sin(tgt_yaw)[:, None], torch.cos(tgt_yaw)[:, None],
    rel(s["bp"]), torch.sin(byaw)[:, None], torch.cos(byaw)[:, None], torch.cos(s["tilt"])[:, None],
    10.0 * s["hook_err"].reshape(n, 6),  # 10 cm で 1（3 mm の合わせを見やすく）
    20.0 * s["gap"][:, None], (top - s["rp"][:, 2])[:, None],
  ], dim=-1)
  manip = (mode > 0)[:, None]
  out[:, 3:] = torch.where(manip, out[:, 3:], torch.zeros_like(out[:, 3:]))
  return out


def command_loco_only(env, command_name: str) -> torch.Tensor:
  c = mdp.generated_commands(env, command_name=command_name)
  return c * (task_mode(env) == 0).float()[:, None]


# ---------------------------------------------------------------- 行動


class ManipArmAction(ArmScheduleAction):
  """歩く課題では V14 と同じ（腕の行動は捨て、外から動かす）。取り上げ・置くでは腕の行動を使う。速度の指令は 0 にする。"""

  def process_actions(self, actions):
    env = self._env
    manip = task_mode(env) > 0
    if self.schedule and self._bank is not None:
      dt = env.step_dt
      self._k += dt
      ch = torch.nonzero(self._k >= self._next).flatten()
      if len(ch):
        self._q_from[ch] = self.arm_target[ch]
        self._q_to[ch] = self._pick(ch, is_empty(env)[ch])
        self._dur[ch] = 0.5 + 1.0 * torch.rand(len(ch), device=self.device)
        self._next[ch] = self._dur[ch] + 0.5 + 2.5 * torch.rand(len(ch), device=self.device)
        self._k[ch] = 0.0
      a = torch.clamp(self._k / self._dur, 0.0, 1.0)
      a = a * a * (3 - 2 * a)
      self.arm_target = self._q_from + (self._q_to - self._q_from) * a[:, None]
    act = actions.clone()
    arm = act[:, self._arm_idx]
    act[:, self._arm_idx] = torch.where(manip[:, None], arm, torch.zeros_like(arm))
    JointPositionAction.process_actions(self, act)
    # 取り上げ・置くの腕：行動 0 の姿勢を課題の始めの構え（取り上げ＝受け具を外へ開いて 1 cm 下、置く＝箱を 3 cm 上げた）にし、
    # 目標をなめらかに追わせる（時定数 約 0.1 秒）。行動 0 が抱える姿勢だと、始めの 0.1 秒で腕が閉じ、受け具が
    # フランジの縁にぶつかって箱を倒した（run16d、目の前から始めた取り上げの 81 % が 1 秒で箱を落とした）
    cur = self._processed_actions[:, self._arm_idx]
    if manip.any() and hasattr(env, "_fg_manip_arm_q"):
      carry_q, ready_q, raise_q = env._fg_manip_arm_q
      base = env._fg_arm_base if hasattr(env, "_fg_arm_base") else torch.where((task_mode(env) == 1)[:, None], ready_q, raise_q)
      # 腕の行動は 1 あたり ARM_ACT_SCALE [rad]（関節によらずそろえる）、±ARM_ACT_CLIP まで。目標の変化は毎秒 ARM_MAX_VEL [rad] まで
      # （上限が無いと、生の行動が ±5〜8 に振れ、歩行の 8 倍の硬さの腕が受け具で箱を上へ放り投げた：run16e、0.28 秒で 63 cm）
      raw = torch.clamp(self._raw_actions[:, self._arm_idx], -ARM_ACT_CLIP, ARM_ACT_CLIP)
      want = base + ARM_ACT_SCALE * raw
      if not hasattr(self, "_arm_filt"):
        self._arm_filt = want.clone()
      step = torch.clamp(ARM_FILTER * (want - self._arm_filt), -ARM_MAX_VEL * env.step_dt, ARM_MAX_VEL * env.step_dt)
      self._arm_filt = torch.where(manip[:, None], self._arm_filt + step, self._arm_filt)
      cur = torch.where(manip[:, None], self._arm_filt, cur)
    if self.arm_target is not None:
      m = self.arm_target_mask & ~manip if self.arm_target_mask is not None else ~manip
      cur = torch.where(m[:, None], self.arm_target, cur)
    self._processed_actions[:, self._arm_idx] = cur
    # 腰の目標を外から固定（統合の置くで使う。親の ArmLockedJointPositionAction の処理を通らないので、ここで同じことをする。
    # 腰を固定しないと、方策が上体を反らせ・傾けて腕の下げを打ち消し、手首が下がらなかった：2026-10-13 17:40）
    if self.waist_target is not None and self.waist_mask is not None:
      jw = self._waist_idx
      self._processed_actions[:, jw] = torch.where(self.waist_mask, self.waist_target, self._processed_actions[:, jw])
    cmd = env.command_manager.get_term("twist")
    cmd.vel_command_b[manip] = 0.0


@dataclass(kw_only=True)
class ManipArmActionCfg(JointPositionActionCfg):
  def build(self, env):
    return ManipArmAction(self, env)


# ---------------------------------------------------------------- リセット


def _arm_q(env):
  """腕の姿勢（行動の腕の並び）：抱える、受け具を構えた（外へ SPREAD、1 cm 下）。"""
  if not hasattr(env, "_fg_manip_arm_q"):
    from fg.arm_ik import ArmIK

    act = env.action_manager.get_term("joint_pos")
    ik = ArmIK()
    out = []
    for kw in ({}, {"spread": SPREAD, "dz": -0.010}, {"dz": PLACE_RAISE}):
      r, _, _ = ik.solve(**kw)
      out.append(torch.tensor([r[nm] for nm in act.arm_names], device=env.device))
    env._fg_manip_arm_q = out
    env._fg_arm_jids = env.scene["robot"].find_joints(tuple(act.arm_names), preserve_order=True)[0]
  return env._fg_manip_arm_q


def _start_offset(k, near_p, dev):
  """寄る距離 [m]：near_p の割合で 0（目の前）、残りは 0.05〜0.3 m と 0.3〜0.7 m を半々"""
  u = torch.rand(k, device=dev)
  mid = 0.05 + 0.25 * torch.rand(k, device=dev)
  far = 0.3 + 0.4 * torch.rand(k, device=dev)
  d = torch.where(u < near_p, torch.zeros(k, device=dev), torch.where(u < near_p + (1 - near_p) / 2, mid, far))
  return d


def reset_task(env, env_ids):
  """課題を割り当て、取り上げ・置くの台はテーブル・箱・ロボットを置き直す（ロボット・箱の既存のリセットのあとに呼ぶ）。"""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  dev = env.device
  mode = task_mode(env)
  k = len(env_ids)
  u = torch.rand(k, device=dev)
  m = torch.where(u < P_MODE[0], 0, torch.where(u < P_MODE[0] + P_MODE[1], 1, 2))
  force = getattr(env, "_fg_force_mode", None)
  if force is None and os.environ.get("FG_FORCE_MODE"):
    force = int(os.environ["FG_FORCE_MODE"])  # 評価用（evaluate_steps.py など、env に触れないスクリプトから）
  if force is not None:
    m = torch.full((k,), int(force), device=dev)  # 評価・動画用：課題を決める
  mode[env_ids] = m
  if hasattr(env, "_fg_succ_t"):
    env._fg_succ_t[env_ids] = 0.0
  if hasattr(env, "_fg_seated"):
    env._fg_seated[env_ids] = False
  if hasattr(env, "_fg_seat_t"):
    env._fg_seat_t[env_ids] = 0.0
  org = env.scene.env_origins[env_ids]
  table = env.scene["table"]
  robot, box = env.scene["robot"], env.scene["bin"]
  # テーブル：歩く台は遠くへ（地面の下）。取り上げ・置くは高さに乱数
  dz = (torch.rand(k, device=dev) * 2 - 1) * TABLE_DZ + torch.where(m == 2, HB - HA, 0.0)
  tz = torch.where(m == 0, torch.full((k,), -3.0, device=dev), dz)
  env._fg_table_top[env_ids] = HA + dz
  tpose = torch.cat([org + torch.stack([torch.full((k,), TX, device=dev), torch.zeros(k, device=dev), tz], dim=-1),
                     torch.tensor([1.0, 0, 0, 0], device=dev).expand(k, 4)], dim=-1)
  table.write_mocap_pose_to_sim(tpose, env_ids=env_ids)
  manip = m > 0
  if not manip.any():
    return
  ids = env_ids[manip]
  mm = m[manip]
  kk = len(ids)
  o = env.scene.env_origins[ids]
  is_empty(env)[ids] = True  # 歩く課題の罰（箱の傾き・ずれ）・落下の判定・腕の予定を、取り上げ・置くの台では使わない（置く課題で手を抜くと「ずれ」の罰になるため）
  carry_q, ready_q, raise_q = _arm_q(env)
  c_root = torch.tensor(layout()["center_root"], dtype=torch.float32, device=dev)
  top = env._fg_table_top[ids]
  # 箱の置き場所（取り上げ）：テーブルの中心に、±1 cm・±3 度
  byaw = torch.where(mm == 1, (torch.rand(kk, device=dev) * 2 - 1) * math.radians(3.0), torch.zeros(kk, device=dev))
  bxy = torch.tensor([TX, 0.0], device=dev) + torch.where((mm == 1)[:, None], (torch.rand(kk, 2, device=dev) * 2 - 1) * 0.01, 0.0)
  # ロボット：箱が抱える位置に来る所（向きは箱に合わせる）から d だけ手前。目の前はずれ小、離れるほどずれ大
  # 寄るのは速度の指令で歩く別のタスクにしたので、目の前（7 割）か少し手前（0.02〜0.15 m）から始める
  u2 = torch.rand(kk, device=dev)
  d = torch.where(u2 < 0.7, torch.zeros(kk, device=dev), 0.02 + 0.13 * torch.rand(kk, device=dev))
  if getattr(env, "_fg_force_d", None) is not None:
    d = torch.full((kk,), float(env._fg_force_d), device=dev)  # 評価・動画用：寄る距離を決める
  near = d == 0
  ryaw = byaw + (torch.rand(kk, device=dev) * 2 - 1) * math.radians(3.0)
  cy, sy = torch.cos(ryaw), torch.sin(ryaw)
  cr = torch.stack([cy * c_root[0] - sy * c_root[1], sy * c_root[0] + cy * c_root[1]], dim=-1)
  lat = (torch.rand(kk, 2, device=dev) * 2 - 1) * 0.03
  rxy = bxy - cr - d[:, None] * torch.stack([cy, sy], dim=-1) + lat
  rz = torch.full((kk,), KNEES_BENT_KEYFRAME.pos[2] + 0.01, device=dev)
  rpose = torch.cat([o + torch.cat([rxy, rz[:, None]], dim=-1), _yaw_quat(ryaw)], dim=-1)
  robot.write_root_link_pose_to_sim(rpose, env_ids=ids)
  robot.write_root_link_velocity_to_sim(torch.zeros(kk, 6, device=dev), env_ids=ids)
  # 関節：既定の姿勢。取り上げの台の腕は、目の前なら構え、離れていれば抱える〜構えの間
  jp = robot.data.default_joint_pos[ids].clone()
  w = torch.ones(kk, device=dev)  # 取り上げの腕は、いつも受け具を構えた姿勢から
  arm = carry_q + w[:, None] * (ready_q - carry_q)
  arm = torch.where((mm == 1)[:, None], arm, raise_q.expand(kk, -1))
  jp[:, env._fg_arm_jids] = arm
  robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
  at = env.action_manager.get_term("joint_pos")
  if not hasattr(at, "_arm_filt"):
    at._arm_filt = carry_q.expand(env.num_envs, -1).clone()
  at._arm_filt[ids] = arm  # 腕の目標のなめらかさの始めの値＝始めの姿勢
  # 箱：取り上げはテーブルの上、置くは手の上
  pick_ids = ids[mm == 1]
  if len(pick_ids):
    sel = mm == 1
    p = o[sel] + torch.cat([bxy[sel], (top[sel] + OZ + 0.001)[:, None]], dim=-1)
    box.write_root_link_pose_to_sim(torch.cat([p, _yaw_quat(byaw[sel])], dim=-1), env_ids=pick_ids)
    box.write_root_link_velocity_to_sim(torch.zeros(len(pick_ids), 6, device=dev), env_ids=pick_ids)
  place_ids = ids[mm == 2]
  if not hasattr(env, "_fg_arm_base"):
    env._fg_arm_base = carry_q.expand(env.num_envs, -1).clone()
  env._fg_arm_base[ids] = torch.where((mm == 1)[:, None], ready_q, raise_q)
  if len(place_ids):
    sel = mm == 2
    # 3 割は「受け具に掛けたまま箱が天板に載った」所から始める（逆向きの練習：手を抜く練習を増やす（2026-10-13）
    seat = torch.rand(len(place_ids), device=dev) < PLACE_SEATED_START
    rise = torch.where(seat, 0.0, PLACE_RAISE)
    off = c_root.expand(len(place_ids), 3) + torch.stack([torch.zeros_like(rise), torch.zeros_like(rise), rise + 0.002], dim=-1)
    if seat.any():
      sid = place_ids[seat]
      jp2 = robot.data.default_joint_pos[sid].clone()
      jp2[:, env._fg_arm_jids] = carry_q
      robot.write_joint_state_to_sim(jp2, torch.zeros_like(jp2), env_ids=sid)
      at._arm_filt[sid] = carry_q
      env._fg_arm_base[sid] = carry_q
      # 天板＝立ったときの抱えた箱の底（胴体が 2 cm 沈む分を引く）
      top_s = rz[sel][seat] + c_root[2] - OZ - 0.02
      env._fg_table_top[sid] = top_s
      tp = torch.cat([env.scene.env_origins[sid] + torch.stack([torch.full_like(top_s, TX), torch.zeros_like(top_s), top_s - HA], dim=-1),
                      torch.tensor([1.0, 0, 0, 0], device=dev).expand(len(sid), 4)], dim=-1)
      table.write_mocap_pose_to_sim(tp, env_ids=sid)
      env._fg_seated[sid] = True
    p = rpose[sel, :3] + quat_apply(rpose[sel, 3:], off)
    box.write_root_link_pose_to_sim(torch.cat([p, rpose[sel, 3:]], dim=-1), env_ids=place_ids)
    box.write_root_link_velocity_to_sim(torch.zeros(len(place_ids), 6, device=dev), env_ids=place_ids)
  # 置く：統合で「製品テーブルに寄って置くに入った瞬間」の状態から始める（2026-10-13 14:40）。
  # 今までの置くの始め（載せる前の 7 割）は、箱が受け具に掛かっておらず 0.1 秒で天板へ落ちていた（学習した置くの成功 29 % は、ほぼ「載せた後」の 3 割）
  pbank = _place_bank(env)
  place_yaw0 = None
  if pbank is not None and len(place_ids):
    sel2 = env._fg_seated[place_ids] == False  # noqa: E712  載せた後から始める台はそのまま
    usep = sel2 & (torch.rand(len(place_ids), device=dev) < PLACE_BANK_P)
    pid = place_ids[usep]
    if len(pid):
      jj = torch.randint(0, pbank["root_pos"].shape[0], (len(pid),), device=dev)
      ob = env.scene.env_origins[pid]
      # 統合（製品テーブルは −x 側、ロボットは −x を向く）→ 学習（テーブルは +x の TX、ロボットは +x を向く）：テーブルの中心まわりに 180 度回す
      qz = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(len(pid), 4)
      c = torch.tensor([TX, 0.0, 0.0], device=dev)

      def rot(v):
        return torch.stack([-v[:, 0], -v[:, 1], v[:, 2]], dim=-1)

      robot.write_root_link_pose_to_sim(torch.cat([ob + c + rot(pbank["root_pos"][jj]), quat_mul(qz, pbank["root_quat"][jj])], dim=-1), env_ids=pid)
      robot.write_root_link_velocity_to_sim(torch.cat([rot(pbank["root_lin_w"][jj]), rot(pbank["root_ang_w"][jj])], dim=-1), env_ids=pid)
      robot.write_joint_state_to_sim(pbank["q"][jj], pbank["qvel"][jj], env_ids=pid)
      at._arm_filt[pid] = pbank["q"][jj][:, env._fg_arm_jids]
      env._fg_arm_base[pid] = raise_q
      env._fg_table_top[pid] = pbank["top"][jj]
      tpp = torch.cat([ob + torch.stack([torch.full((len(pid),), TX, device=dev), torch.zeros(len(pid), device=dev), pbank["top"][jj] - HA], dim=-1),
                       torch.tensor([1.0, 0, 0, 0], device=dev).expand(len(pid), 4)], dim=-1)
      table.write_mocap_pose_to_sim(tpp, env_ids=pid)
      box.write_root_link_pose_to_sim(torch.cat([ob + c + rot(pbank["box_pos"][jj]), quat_mul(qz, pbank["box_quat"][jj])], dim=-1), env_ids=pid)
      box.write_root_link_velocity_to_sim(torch.cat([rot(pbank["box_lin_w"][jj]), rot(pbank["box_ang_w"][jj])], dim=-1), env_ids=pid)
      place_yaw0 = (pid, _yaw(quat_mul(qz, pbank["root_quat"][jj])))
  # 統合で「歩いて寄った後」に取り上げへ入った瞬間の状態から始める（2026-10-13）。
  # 学習の始めが静止した既定の姿勢だけだと、統合の中の取り上げで 2 割前後が転んだ
  bank = _entry_bank(env)
  bid = pick_ids[:0]
  j = None
  if bank is not None and len(pick_ids):
    use = torch.rand(len(pick_ids), device=dev) < ENTRY_BANK_P
    if getattr(env, "_fg_force_bank", None) is not None:
      use = torch.full_like(use, bool(env._fg_force_bank))
    bid = pick_ids[use]
    if len(bid):
      j = torch.randint(0, bank["root_pos"].shape[0], (len(bid),), device=dev)
      ob = env.scene.env_origins[bid]
      robot.write_root_link_pose_to_sim(torch.cat([ob + bank["root_pos"][j], bank["root_quat"][j]], dim=-1), env_ids=bid)
      robot.write_root_link_velocity_to_sim(torch.cat([bank["root_lin_w"][j], bank["root_ang_w"][j]], dim=-1), env_ids=bid)
      robot.write_joint_state_to_sim(bank["q"][j], bank["qvel"][j], env_ids=bid)
      at._arm_filt[bid] = bank["q"][j][:, env._fg_arm_jids]
      # テーブルは統合と同じ高さ・位置、箱は記録した位置
      env._fg_table_top[bid] = HA
      tpb = torch.cat([ob + torch.tensor([TX, 0.0, 0.0], device=dev), torch.tensor([1.0, 0, 0, 0], device=dev).expand(len(bid), 4)], dim=-1)
      table.write_mocap_pose_to_sim(tpb, env_ids=bid)
      box.write_root_link_pose_to_sim(torch.cat([ob + bank["box_pos"][j], bank["box_quat"][j]], dim=-1), env_ids=bid)
      box.write_root_link_velocity_to_sim(torch.zeros(len(bid), 6, device=dev), env_ids=bid)
  # 取り上げの始めの胴体の向き（取り上げ中に体が回るのを抑える報酬・成功の条件に使う。2026-10-13）
  if not hasattr(env, "_fg_yaw0"):
    env._fg_yaw0 = torch.zeros(env.num_envs, device=dev)
  env._fg_yaw0[ids] = ryaw
  if bank is not None and len(pick_ids) and len(bid):
    env._fg_yaw0[bid] = _yaw(bank["root_quat"][j])
  if place_yaw0 is not None:
    env._fg_yaw0[place_yaw0[0]] = place_yaw0[1]
  # 腕の行動の最初の値（前回の行動の入力）は 0 のまま。腕の予定は使わない
  env._fg_ts_step = None


ENTRY_BANK_P = float(os.environ.get("FG_ENTRY_BANK_P", "0.5"))
PLACE_BANK_P = float(os.environ.get("FG_PLACE_BANK_P", "1.0"))


def _place_bank(env):
  """FG_PLACE_BANK（integrate.py --entry-log の .pt の entry_place、複数は ; 区切り）を読む（1 回だけ）。"""
  if hasattr(env, "_fg_place_bank"):
    return env._fg_place_bank
  env._fg_place_bank = None
  paths = os.environ.get("FG_PLACE_BANK", "")
  if not paths:
    return None
  ents = []
  for pth in paths.split(";"):
    d = torch.load(pth, weights_only=False)
    ents += [e for e in d.get("entry_place", []) if e is not None]
  if not ents:
    return None
  dev = env.device
  keys = ("root_pos", "root_quat", "root_lin_w", "root_ang_w", "q", "qvel", "box_pos", "box_quat", "box_lin_w", "box_ang_w", "top")
  env._fg_place_bank = {k: torch.tensor([e[k] for e in ents], dtype=torch.float32, device=dev) for k in keys}
  print(f"[pick_learn] 置くの始めの場面：統合の記録 {len(ents)} 件（載せる前の台の {PLACE_BANK_P:.0%}）")
  return env._fg_place_bank


def _entry_bank(env):
  """FG_ENTRY_BANK（integrate.py --entry-log の .pt、複数は ; 区切り）から、取り上げに入った瞬間の状態を読む（1 回だけ）。"""
  if hasattr(env, "_fg_entry_bank"):
    return env._fg_entry_bank
  env._fg_entry_bank = None
  paths = os.environ.get("FG_ENTRY_BANK", "")
  if not paths:
    return None
  ents = []
  for pth in paths.split(";"):
    d = torch.load(pth, weights_only=False)
    ents += [e for e in d["entry"] if e is not None and "qvel" in e]
  if not ents:
    return None
  dev = env.device
  env._fg_entry_bank = {k: torch.tensor([e[k] for e in ents], dtype=torch.float32, device=dev)
                        for k in ("root_pos", "root_quat", "root_lin_w", "root_ang_w", "q", "qvel", "box_pos", "box_quat")}
  print(f"[pick_learn] 取り上げの始めの場面：統合の記録 {len(ents)} 件を {ENTRY_BANK_P:.0%} の割合で使う")
  return env._fg_entry_bank


# ---------------------------------------------------------------- 報酬・終了


def _m(env, k):
  return (task_mode(env) == k).float()


def loco_only(func):
  """報酬を歩く課題の台だけにする（関数でも、mjlab の報酬のクラス（variable_posture など）でもよい）。"""
  if isinstance(func, type):
    class Wrapped(func):
      def __call__(self, env, **kw):
        return super().__call__(env, **kw) * _m(env, 0)

    Wrapped.__name__ = func.__name__ + "_loco"
    return Wrapped

  def f(env, **kw):
    return func(env, **kw) * _m(env, 0)

  f.__name__ = func.__name__ + "_loco"
  return f


def manip_only(func):
  """報酬のクラスを、取り上げ・置くの台だけにする。"""
  class Wrapped(func):
    def __call__(self, env, **kw):
      return super().__call__(env, **kw) * (task_mode(env) > 0).float()

  Wrapped.__name__ = func.__name__ + "_manip"
  return Wrapped


def pick_reach(env) -> torch.Tensor:
  """受け板が、フランジの下の決まった位置へ近いほど 1（3 cm で 1/e）。"""
  s = _cached(env)
  return torch.exp(-s["hook_dist"] / 0.03) * _m(env, 1)


def pick_approach(env) -> torch.Tensor:
  """受け板を箱へ寄せる（遠くからの引き寄せ：25 cm で 1/e）。pick_reach（3 cm）だけでは、0.3〜0.7 m 離れて始めた台に
  手がかりがなく、run16b は箱を落とす罰を避けて、テーブルから離れて立つことを覚えた"""
  s = _cached(env)
  return torch.exp(-s["hook_dist"] / 0.25) * _m(env, 1)


def place_approach(env) -> torch.Tensor:
  """箱をテーブルの中心の上へ寄せる（遠くからの引き寄せ：30 cm で 1/e）。受け具に載せている間だけ。"""
  s = _cached(env)
  return torch.exp(-s["dxy"] / 0.30) * (s["hook_dist"] < 0.03).float() * (~env._fg_seated).float() * _m(env, 2)


def _engaged(s):
  return s["hook_dist"] < 0.02


# 取り上げで向きを保つ（FG_PICK_YAW=1 のときだけ、報酬・成功の条件に入れる。run19 までは無し）
PICK_YAW = os.environ.get("FG_PICK_YAW", "") == "1"
# 成功の条件に向きを入れるか（run20 は入れて、最初から成功が取れず、0.3 秒で箱を落として早く終わらせる方策に崩れた：2026-10-13 13:10）
PICK_YAW_SUCCESS = os.environ.get("FG_PICK_YAW_SUCCESS", "") == "1"
PICK_YAW_TOL = float(os.environ.get("FG_PICK_YAW_TOL", "10"))  # [度]
PICK_WZ_TOL = float(os.environ.get("FG_PICK_WZ_TOL", "0.2"))  # [rad/s]
# 取り上げの成功・持ち上げのご褒美を、箱がテーブルの上（中心からこの距離以内）にあるときだけにする（0 で無効）。
# 今の取り上げの方策は、箱を天板の端から引きずり出して「浮いた」ことにしていた（統合で 10 cm 以内を求めると成功が 1 割）：2026-10-13 19:45
PICK_DXY = float(os.environ.get("FG_PICK_DXY", "0"))
# 置くでも向きを保つご褒美（学習した置くも体を 15 度ほど回し、統合では箱を持ち上げて傾けた：2026-10-13 16:40）
PLACE_YAW = os.environ.get("FG_PLACE_YAW", "") == "1"


def _pick_yaw_err(env, s):
  yaw0 = env._fg_yaw0 if hasattr(env, "_fg_yaw0") else torch.zeros(env.num_envs, device=env.device)
  return torch.remainder(_yaw(s["rq"]) - yaw0 + math.pi, 2 * math.pi) - math.pi


def pick_yaw(env) -> torch.Tensor:
  """取り上げ中に向きを保っているほど 1（ずれ 0.3 rad ≒ 17 度で 1/e）。統合で、取り上げの最中に体が 30〜60 度回って箱を落としていた（2026-10-13）。
  罰（−ずれ²）にした run20 は、罰を避けて早く終わる（箱を落とす）方策に崩れたので、保てば得をするご褒美にする"""
  s = _cached(env)
  m = _m(env, 1) + (_m(env, 2) if PLACE_YAW else 0.0)
  return torch.exp(-((_pick_yaw_err(env, s) / 0.3) ** 2)) * m


def pick_yaw_rate(env) -> torch.Tensor:
  """取り上げ中の胴体の向きの角速度の 2 乗（罰）。"""
  wz = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
  m = _m(env, 1) + (_m(env, 2) if PLACE_YAW else 0.0)
  return wz**2 * m


def pick_lift(env) -> torch.Tensor:
  """差し込んだまま箱が天板から浮いた高さ（4 cm で 1）。傾き 15 度以上は 0。"""
  s = _cached(env)
  ok = _engaged(s) & (s["tilt"] < math.radians(15.0))
  if PICK_DXY > 0:
    ok = ok & (s["dxy"] < PICK_DXY)
  return torch.clamp(s["gap"] / 0.04, 0.0, 1.0) * ok.float() * _m(env, 1)


def pick_away(env) -> torch.Tensor:
  """持ち上げた（2 cm 以上）箱を、テーブルの中心から水平に離した距離（0.4 m で 1）。"""
  s = _cached(env)
  ok = _engaged(s) & (s["gap"] > 0.02) & (s["tilt"] < math.radians(15.0)) & s["standing"]
  return torch.clamp(s["dxy"] / 0.4, 0.0, 1.0) * ok.float() * _m(env, 1)


def _resting(s):
  return (s["gap"].abs() < 0.006) & (s["tilt"] < math.radians(6.0)) & (s["dxy"] < 0.08)


def place_over(env) -> torch.Tensor:
  """箱をテーブルの中心の上へ（水平 5 cm で 1/e）。箱を受け具に載せている間だけ。"""
  s = _cached(env)
  return torch.exp(-s["dxy"] / 0.05) * (s["hook_dist"] < 0.03).float() * (~env._fg_seated).float() * _m(env, 2)


def place_down(env) -> torch.Tensor:
  """テーブルの上（水平 6 cm 以内）で、箱の底を天板へ（1 cm で 1/e）。載ったら 1。"""
  s = _cached(env)
  over = s["dxy"] < 0.06
  # 受け具に掛けたまま下ろしたとき（または、ゆっくり載せた後）だけ（run16f は腕を上げて箱を落とし、この報酬だけ取っていた）
  hooked = (s["hook_dist"] < 0.03) & ~env._fg_seated
  return torch.exp(-torch.clamp(s["gap"], min=0.0) / 0.01) * over.float() * hooked.float() * (s["tilt"] < math.radians(10.0)).float() * _m(env, 2)


def place_seated(env) -> torch.Tensor:
  """ゆっくり載せた後、箱が天板に載って立っている間 1（手を抜いても減らない）。"""
  s = _cached(env)
  return (env._fg_seated & _resting(s) & s["standing"]).float() * _m(env, 2)


def _update_seated(env, s):
  """（終了の判定から 1 ステップに 1 回呼ぶ）受け具に掛けたまま、箱が天板にゆっくり載ったら記録する。"""
  slow = torch.linalg.norm(s["bv"], dim=-1) < 0.08
  # 受け板の平均のずれ 5 cm 以内（両方 3 cm 以内では、載せた後の受け板のずれ 18〜38 mm で判定が付かなかった：run16h）
  both = s["hook_dist"] < 0.05
  ok = (s["mode"] == 2) & _resting(s) & slow & both
  if not hasattr(env, "_fg_seat_t"):
    env._fg_seat_t = torch.zeros(env.num_envs, device=env.device)
  env._fg_seat_t = torch.where(ok, env._fg_seat_t + env.step_dt, torch.zeros_like(env._fg_seat_t))
  env._fg_seated |= env._fg_seat_t >= 0.06


def _released(s):
  """両手を外へ抜いた：受け板が、左は +y、右は −y へ 5 cm 以上（腕を上げただけでは数えない）。"""
  e = s["hook_err"]
  return (e[:, 0, 1] > 0.05) & (e[:, 1, 1] < -0.05)


def place_release(env) -> torch.Tensor:
  """箱が天板に載ったら、受け板をフランジの下から抜く（5 cm で 1）。"""
  s = _cached(env)
  e = s["hook_err"]
  l, r = e[:, 0, 1], -e[:, 1, 1]
  # 外へまっすぐ抜く：前後・上下にずらすほど減らす（run16h は片手を後ろ・下へ引き、もう片方が掛かったまま箱を 41 度傾けて落とした）
  sl = torch.exp(-(e[:, 0, 0].abs() + e[:, 0, 2].abs()) / 0.03)
  sr = torch.exp(-(e[:, 1, 0].abs() + e[:, 1, 2].abs()) / 0.03)
  out = 0.5 * (torch.clamp(l / 0.05, 0.0, 1.0) * sl + torch.clamp(r / 0.05, 0.0, 1.0) * sr)
  return out * (_resting(s) & s["standing"] & env._fg_seated).float() * _m(env, 2)


def _root_box_dist(s):
  return torch.linalg.norm(s["rp"][:, :2] - s["bp"][:, :2], dim=-1)


def place_away(env) -> torch.Tensor:
  """載せて手を抜いたら、胴体を箱から離す（抱えたときの約 0.36 m から ＋0.25 m で 1）。"""
  s = _cached(env)
  ok = _resting(s) & _released(s) & s["standing"] & env._fg_seated
  return torch.clamp((_root_box_dist(s) - 0.36) / 0.25, 0.0, 1.0) * ok.float() * _m(env, 2)


SUCCESS_HOLD = 0.5  # [s] 成功の状態がこれだけ続いたら成功
PICK_HOLD = 1.0  # [s] 取り上げは持ち上げたまま 1 秒保ったら成功


def task_success(env) -> torch.Tensor:
  """成功（終了）：取り上げ＝箱を 2 cm 以上持ち上げ、テーブルの中心から 0.35 m 離した。置く＝箱が天板に載り、受け板を 5 cm 抜き、胴体が箱から 0.55 m 離れた。
  どちらも立ったまま、その状態が SUCCESS_HOLD 秒続いたら"""
  s = _cached(env)
  mode = s["mode"]
  # 個別のタスクに分けた（2026-10-13 1:30、作者の決定：個別のタスクを 100 % にしてから統合）。取り上げは「持ち上げて 1 秒保つ」まで、
  # 置くは「ゆっくり載せて両手を抜く」まで。テーブルから離れる・下がるのは、速度の指令で歩く別のタスク（T4・T8）にする
  pick_ok = (mode == 1) & _engaged(s) & (s["gap"] > 0.02) & (s["tilt"] < math.radians(10.0))
  if PICK_DXY > 0:
    pick_ok = pick_ok & (s["dxy"] < PICK_DXY)
  if PICK_YAW_SUCCESS:
    # 向きを保ったまま持ち上げたことも成功の条件に（箱と体が一緒に回っても成功になっていた）
    wz = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
    pick_ok = pick_ok & (_pick_yaw_err(env, s).abs() < math.radians(PICK_YAW_TOL)) & (wz.abs() < PICK_WZ_TOL)
  _update_seated(env, s)
  place_ok = (mode == 2) & _resting(s) & _released(s) & env._fg_seated
  ok = (pick_ok | place_ok) & s["standing"]
  if not hasattr(env, "_fg_succ_t"):
    env._fg_succ_t = torch.zeros(env.num_envs, device=env.device)
  env._fg_succ_t = torch.where(ok, env._fg_succ_t + env.step_dt, torch.zeros_like(env._fg_succ_t))
  return env._fg_succ_t >= torch.where(mode == 1, PICK_HOLD, SUCCESS_HOLD)


def fell_penalty(env) -> torch.Tensor:
  """転倒したステップで 1（取り上げ・置くの台。歩く台は今までどおり罰なし）。"""
  return env.termination_manager.get_term("fell_over").float() * (task_mode(env) > 0).float()


def success_bonus(env) -> torch.Tensor:
  return env.termination_manager.get_term("task_success").float()


def manip_drop(env) -> torch.Tensor:
  """取り上げ・置くの台で箱を落とした（終了）ステップで 1。"""
  return env.termination_manager.get_term("bin_dropped").float() * (task_mode(env) > 0).float()


def manip_bin_vel(env) -> torch.Tensor:
  """箱の速さの二乗 [(m/s)^2]（取り上げ・置くの台。箱を振り回す・投げるのを防ぐ）。"""
  s = _cached(env)
  return torch.sum(torch.square(s["bv"]), dim=-1) * (task_mode(env) > 0).float()


def action_rate_masked(env) -> torch.Tensor:
  """行動の変化の二乗の和。取り上げ・置くの台の腕は 0.1 倍（腕は目標の速さの上限となめらかさで別に抑えている）。"""
  am = env.action_manager
  d = torch.square(am.action - am.prev_action)
  at = am.get_term("joint_pos")
  w = torch.ones_like(d)
  manip = task_mode(env) > 0
  w[:, at._arm_idx] = torch.where(manip[:, None], 0.1, 1.0)
  return torch.sum(d * w, dim=-1)


def manip_press(env) -> torch.Tensor:
  """箱を天板に押し込んだ深さ [m]（2 mm を超えた分。テーブルの上だけ）。押し込むと、離した瞬間に箱が跳ねる。"""
  s = _cached(env)
  over = s["dxy"] < 0.10
  return torch.clamp(-s["gap"] - 0.002, min=0.0) * over.float() * (task_mode(env) > 0).float()


def manip_time(env) -> torch.Tensor:
  return (task_mode(env) > 0).float()


def manip_bin_tilt(env) -> torch.Tensor:
  s = _cached(env)
  return torch.square(s["tilt"]) * (task_mode(env) > 0).float()


def bin_dropped_task(env) -> torch.Tensor:
  """箱を落とした（終了）：歩く課題は V13 と同じ（受け具から 4 cm ずれた）。取り上げ・置くは、箱が天板より 10 cm 下へ落ちた・45 度以上傾いた。"""
  s = _cached(env)
  mode = s["mode"]
  loco = bin_dropped_carrying(env) & (mode == 0)
  # 投げ上げ：箱の底が天板より 12 cm 以上上（持ち上げは 2〜4 cm）。最初は「上へ 0.5 m/s 以上」にしたが、腕で箱を天板に押し込み
  # （めり込み 5 mm）、離した瞬間に跳ねる 0.5 m/s も落下にしていた（run16h の置くの落下 81 %、2026-10-13 2:05）
  fell = (s["bp"][:, 2] - OZ < s["top"] - 0.10) | (s["tilt"] > math.radians(45.0)) | (s["gap"] > 0.12)
  return loco | ((mode > 0) & fell)


# ---------------------------------------------------------------- 設定


def luna_pick_place_v16_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = luna_carry_steps_v14_env_cfg(play=play)
  # 床は平ら（取り上げ・置くの場所は平らな床。段差は run9〜run14 の課題で扱う）
  cfg.curriculum.pop("terrain_levels", None)
  if V16_STEPS:
    from dataclasses import replace as _rep

    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    from fg.terrains import STEP_HEIGHT_RANGE, TILE_SIZE, TWO_STEP_PLATFORM

    lo, hi = STEP_HEIGHT_RANGE
    d = (0.20 - lo) / (hi - lo)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(TILE_SIZE, TILE_SIZE), border_width=4.0, num_rows=6, num_cols=6, curriculum=False,
      sub_terrains={"two_step": _rep(TWO_STEP_PLATFORM, proportion=1.0)}, difficulty_range=(d, d))
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
  if not V16_STEPS:
    cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.events.pop("randomize_terrain", None)
  # テーブル（1 台。課題ごとに高さ・場所を変える）
  cfg.scene.entities["table"] = EntityCfg(spec_fn=table_spec(HA, "table"), init_state=EntityCfg.InitialStateCfg(pos=(TX, 0.0, -3.0)))
  if cfg.sim.nconmax is not None:
    cfg.sim.nconmax += 20
  # 行動
  a = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = ManipArmActionCfg(**{f.name: getattr(a, f.name) for f in fields(a) if f.init})
  # リセット：既存（胴体・関節・箱・持つ／持たない）のあとに課題を割り当てる
  cfg.events["task_mode"] = EventTermCfg(func=reset_task, mode="reset", params={})
  # 評価用：歩く課題で箱を持たない台の割合（既定 0.3 は学習と同じ。FG_EMPTY_FRACTION=0 で全台が箱を持つ、1 で全台が手ぶら）
  cfg.events["carry_mode"].params["fraction"] = float(os.environ.get("FG_EMPTY_FRACTION", "0.3"))
  # 入力：速度の指令は歩く課題だけ。最後に課題の入力を足す（run14 の 280 個の並びはそのまま）
  for g in ("actor", "critic"):
    terms = cfg.observations[g].terms
    terms["command"] = ObservationTermCfg(func=command_loco_only, params={"command_name": "twist"})
    terms["task"] = ObservationTermCfg(func=task_obs)
  # 歩く課題の報酬は、歩く台だけ
  for nm in ("track_linear_velocity", "track_angular_velocity", "lin_vel_error", "ang_vel_error", "stall", "pose"):
    if nm in cfg.rewards:
      cfg.rewards[nm].func = loco_only(cfg.rewards[nm].func)
  # 取り上げ・置くの台の姿勢：脚・腰は歩くときの許容、腕・手首は動かすので緩く（指令が 0 なので「立っている」の許容を使う）
  import copy

  pose = copy.deepcopy(cfg.rewards["pose"])
  std = dict(pose.params["std_walking"])
  std.update({k: 1.0 for k in std if any(a in k for a in ("shoulder", "elbow", "wrist"))})
  pose.params["std_standing"] = std
  pose.func = manip_only(mdp.variable_posture)
  cfg.rewards["manip_pose"] = pose
  # 取り上げ・置くの報酬（報酬は ×dt される）
  cfg.rewards["pick_approach"] = RewardTermCfg(func=pick_approach, weight=1.5, params={})
  cfg.rewards["pick_reach"] = RewardTermCfg(func=pick_reach, weight=3.0, params={})
  cfg.rewards["pick_lift"] = RewardTermCfg(func=pick_lift, weight=3.0, params={})
  if PICK_YAW:
    cfg.rewards["pick_yaw"] = RewardTermCfg(func=pick_yaw, weight=2.0, params={})
    cfg.rewards["pick_yaw_rate"] = RewardTermCfg(func=pick_yaw_rate, weight=-0.1, params={})
  cfg.rewards["place_approach"] = RewardTermCfg(func=place_approach, weight=1.5, params={})
  cfg.rewards["place_over"] = RewardTermCfg(func=place_over, weight=2.0, params={})
  cfg.rewards["place_down"] = RewardTermCfg(func=place_down, weight=2.0, params={})
  cfg.rewards["place_seated"] = RewardTermCfg(func=place_seated, weight=2.0, params={})
  cfg.rewards["place_release"] = RewardTermCfg(func=place_release, weight=3.0, params={})
  cfg.rewards["success"] = RewardTermCfg(func=success_bonus, weight=1000.0, params={})  # 1 回 ＋20
  # 箱を落とす罰：歩く台は V11 のまま（−10）。取り上げ・置くの台は −2（−10 では、run16b・run16c が箱に触れること自体を避け、
  # 受け板を寄せる報酬が学習の途中で 0 へ下がった）
  cfg.rewards["bin_drop"].func = loco_only(cfg.rewards["bin_drop"].func)
  cfg.rewards["manip_drop"] = RewardTermCfg(func=manip_drop, weight=-100.0, params={})
  cfg.rewards["manip_fell"] = RewardTermCfg(func=fell_penalty, weight=-1000.0, params={})  # 1 回 −20（成功と同じ大きさ）
  cfg.rewards["manip_press"] = RewardTermCfg(func=manip_press, weight=-100.0, params={})  # 5 mm で −0.3/秒
  cfg.rewards["manip_bin_vel"] = RewardTermCfg(func=manip_bin_vel, weight=-2.0, params={})
  cfg.rewards["action_rate_l2"].func = action_rate_masked
  cfg.rewards["manip_time"] = RewardTermCfg(func=manip_time, weight=-0.5, params={})
  cfg.rewards["manip_bin_tilt"] = RewardTermCfg(func=manip_bin_tilt, weight=-2.0, params={})
  # 終了
  cfg.terminations["bin_dropped"] = TerminationTermCfg(func=bin_dropped_task, params={})
  cfg.terminations["task_success"] = TerminationTermCfg(func=task_success, params={})
  return cfg


def luna_pick_place_ppo_runner_cfg():
  cfg = luna_steps_ppo_runner_cfg()
  cfg.max_iterations = 3000
  return cfg


register_mjlab_task(
  task_id=PICK_TASK_ID,
  env_cfg=luna_pick_place_v16_env_cfg(),
  play_env_cfg=luna_pick_place_v16_env_cfg(play=True),
  rl_cfg=luna_pick_place_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
