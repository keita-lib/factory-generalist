"""落ちる回収箱を持って、2 段のステップを上り下りする学習（技術成果 3・4、mjlab 1.6.0）。

土台は run9 の課題（V9）。違い：
- 回収箱は手首に固定せず、手首の受け具（コの字の溝）に載せるだけ（fg.free_bin）。落としたら終了＋大きな罰
- 腕は抱える姿勢に固定（方策の出す腕の行動を 0 に置き換える。行動の数は run9 と同じなので、run9 の重みから続けて学習できる）
- 箱の傾き・ずれに小さな罰（缶がこぼれない・揺らさない）
"""

from dataclasses import dataclass, fields

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from fg.free_bin import apply_free_bin, bin_diag, bin_offset_from_hand
from fg.velocity_steps import luna_steps_ppo_runner_cfg, luna_steps_v9_env_cfg

CARRY_V11_TASK_ID = "FG-Carry-Steps-Luna-V11"
ARM_KEYS = ("shoulder", "elbow", "wrist")  # 手首 yaw も固定（受け具がねじれて箱が傾くため。2026-10-12 朝に気づいて追加）
CANS = 3


class ArmLockedJointPositionAction(JointPositionAction):
  """腕（肩・肘・手首 yaw）の行動を 0 にする（目標角は既定の姿勢＝抱える姿勢のまま）。

  arm_target（台数 × 腕の関節数、絶対角 [rad]）を入れると、腕の目標角をそれに置き換える（取り上げ・置く動作で使う）。
  arm_target_mask（台数）が False の台は既定の姿勢のまま
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._arm_idx = torch.tensor([i for i, n in enumerate(self._target_names) if any(a in n for a in ARM_KEYS)],
                                 device=self.device, dtype=torch.long)
    self.arm_names = [self._target_names[i] for i in self._arm_idx.tolist()]
    self.arm_target: torch.Tensor | None = None
    self.arm_target_mask: torch.Tensor | None = None
    # 腰の前後の傾き（waist_pitch）の目標を外から固定する（置く動作で、方策が上体を後ろへ反らせて腕の下げを打ち消すため）
    # 腰の 3 軸（yaw・roll・pitch）の目標を外から固定する（取り上げで、方策が腰を 8 度ねじって上体の向きが胴体とずれたため。
    # waist_target は 台数 × 3、waist_mask は 台数 × 3）
    self._waist_idx = torch.tensor([next(i for i, nm in enumerate(self._target_names) if f"waist_{a}" in nm) for a in ("yaw", "roll", "pitch")],
                                   device=self.device, dtype=torch.long)
    self.waist_target: torch.Tensor | None = None
    self.waist_mask: torch.Tensor | None = None

  def process_actions(self, actions: torch.Tensor):
    actions = actions.clone()
    actions[:, self._arm_idx] = 0.0
    super().process_actions(actions)
    if self.arm_target is not None:
      cur = self._processed_actions[:, self._arm_idx]
      m = self.arm_target_mask[:, None] if self.arm_target_mask is not None else True
      self._processed_actions[:, self._arm_idx] = torch.where(m, self.arm_target, cur)
    if self.waist_target is not None and self.waist_mask is not None:
      j = self._waist_idx
      self._processed_actions[:, j] = torch.where(self.waist_mask, self.waist_target, self._processed_actions[:, j])


@dataclass(kw_only=True)
class ArmLockedJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env):
    return ArmLockedJointPositionAction(self, env)


def bin_drop_penalty(env) -> torch.Tensor:
  """箱を落としたステップで 1。"""
  return env.termination_manager.get_term("bin_dropped").float()


def bin_tilt_penalty(env) -> torch.Tensor:
  """箱の傾きの二乗 [rad^2]。"""
  return torch.square(torch.deg2rad(bin_diag(env)[:, 2]))


def bin_offset_penalty(env) -> torch.Tensor:
  """箱の、抱える位置からのずれの二乗 [m^2]。"""
  return torch.square(bin_offset_from_hand(env))


def luna_carry_steps_v11_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = luna_steps_v9_env_cfg(play=play)
  apply_free_bin(cfg, CANS)
  a = cfg.actions["joint_pos"]
  assert isinstance(a, JointPositionActionCfg)
  cfg.actions["joint_pos"] = ArmLockedJointPositionActionCfg(**{f.name: getattr(a, f.name) for f in fields(a) if f.init})
  # 落としたら大きな罰（転倒と同じく終了。報酬は ×dt される → 1 回 −10 相当）
  cfg.rewards["bin_drop"] = RewardTermCfg(func=bin_drop_penalty, weight=-500.0, params={})
  cfg.rewards["bin_tilt"] = RewardTermCfg(func=bin_tilt_penalty, weight=-2.0, params={})
  cfg.rewards["bin_offset"] = RewardTermCfg(func=bin_offset_penalty, weight=-100.0, params={})
  return cfg


register_mjlab_task(
  task_id=CARRY_V11_TASK_ID,
  env_cfg=luna_carry_steps_v11_env_cfg(),
  play_env_cfg=luna_carry_steps_v11_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


CARRY_V12_TASK_ID = "FG-Carry-Steps-Luna-V12"


def luna_carry_steps_v12_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """V11 に「ゆっくり・横・その場の旋回を正確に」を足した版（テーブルの前で位置を合わせるため）。

  run9 の方策は、0.06〜0.15 m/s の前後・横の指令や 0.1〜0.25 rad/s の旋回の指令では足を出さず、
  テーブルの前で 3〜4 cm・数度のずれが残った（2026-10-12 朝、pick_place.py の試行）。
  1. 指令の範囲を狭めて小さな指令を多く：前後 −0.3〜0.5、横 −0.25〜0.25 m/s、旋回 −0.6〜0.6 rad/s
  2. 速度の追従のご褒美を鋭く（std 0.5 → 0.25）、速度の誤差の罰を −1 → −2、旋回の誤差の罰 −0.3
  3. 立ち止まりの罰を小さな指令にも（指令 0.05 m/s 以上で、その半分未満の速さなら罰）
  """
  from fg.velocity_steps import ang_vel_error_penalty

  cfg = luna_carry_steps_v11_env_cfg(play=play)
  tw = cfg.commands["twist"]
  tw.ranges.lin_vel_x = (-0.3, 0.5)
  tw.ranges.lin_vel_y = (-0.25, 0.25)
  tw.ranges.ang_vel_z = (-0.6, 0.6)
  cfg.rewards["track_linear_velocity"].params["std"] = 0.25
  cfg.rewards["lin_vel_error"].weight = -2.0
  cfg.rewards["ang_vel_error"] = RewardTermCfg(func=ang_vel_error_penalty, weight=-0.3, params={"command_name": "twist"})
  cfg.rewards["stall"].params.update({"min_command": 0.05, "floor": 0.025})
  return cfg


register_mjlab_task(
  task_id=CARRY_V12_TASK_ID,
  env_cfg=luna_carry_steps_v12_env_cfg(),
  play_env_cfg=luna_carry_steps_v12_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


CARRY_V13_TASK_ID = "FG-Carry-Steps-Luna-V13"
EMPTY_FRACTION = 0.3  # 箱を持たない（手を外へ開いて構える）台の割合


def _arm_poses(env):
  """抱える姿勢と、手を外へ 10 cm 開いた構え（取り上げの前）の腕の関節角（行動の腕の並び）。"""
  if not hasattr(env, "_fg_arm_poses"):
    from fg.arm_ik import ArmIK

    act = env.action_manager.get_term("joint_pos")
    ik = ArmIK()
    out = []
    for kw in ({}, {"spread": 0.10}):
      r, _, _ = ik.solve(**kw)
      out.append(torch.tensor([r[nm] for nm in act.arm_names], device=env.device))
    env._fg_arm_poses = out
  return env._fg_arm_poses


def is_empty(env) -> torch.Tensor:
  if not hasattr(env, "_fg_empty"):
    env._fg_empty = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env._fg_empty


def randomize_carry_mode(env, env_ids, fraction: float = EMPTY_FRACTION):
  """リセットのたびに「箱を持つ／持たない」を決める。持たない台は、箱を区画の隅へ置き、腕を抱える姿勢〜開いた構えの間にする。
  （ロボットの・箱のリセットのあとに呼ぶ）"""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  empty = is_empty(env)
  e = torch.rand(len(env_ids), device=env.device) < fraction
  empty[env_ids] = e
  act = env.action_manager.get_term("joint_pos")
  carry_q, spread_q = _arm_poses(env)
  if act.arm_target is None:
    act.arm_target = carry_q.expand(env.num_envs, -1).clone()
    act.arm_target_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  u = torch.rand(len(env_ids), 1, device=env.device)
  act.arm_target[env_ids] = carry_q + u * (spread_q - carry_q)
  act.arm_target_mask[env_ids] = e
  ids = env_ids[e]
  if len(ids):
    box = env.scene["bin"]
    from fg.terrains import TILE_SIZE

    p = env.scene.env_origins[ids] + torch.tensor([TILE_SIZE / 2 - 0.4, TILE_SIZE / 2 - 0.4, 1.0], device=env.device)
    q = torch.tensor([1.0, 0, 0, 0], device=env.device).expand(len(ids), 4)
    box.write_root_link_pose_to_sim(torch.cat([p, q], dim=-1), env_ids=ids)
    box.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def bin_dropped_carrying(env) -> torch.Tensor:
  from fg.free_bin import bin_dropped

  return bin_dropped(env) & ~is_empty(env)


def _masked(func):
  def f(env, **kw):
    return func(env, **kw) * (~is_empty(env)).float()

  f.__name__ = func.__name__ + "_carrying"
  return f


def luna_carry_steps_v13_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """V12 に「箱を持たずに、手を開いて構えたまま歩く」を混ぜた版（3 割の台）。

  取り上げの前は、手を外へ 10 cm 開いた構えで、箱を持たずにテーブルへ近づく。V12 の方策（箱を持つ練習だけ）では、
  この構えで 16 台中 12 台が転んだ（2026-10-12 10 時、pick_place.py）。
  """
  from mjlab.managers.termination_manager import TerminationTermCfg

  cfg = luna_carry_steps_v12_env_cfg(play=play)
  cfg.events["carry_mode"] = EventTermCfg(func=randomize_carry_mode, mode="reset", params={})
  cfg.terminations["bin_dropped"] = TerminationTermCfg(func=bin_dropped_carrying, params={})
  cfg.rewards["bin_tilt"] = RewardTermCfg(func=_masked(bin_tilt_penalty), weight=-2.0, params={})
  cfg.rewards["bin_offset"] = RewardTermCfg(func=_masked(bin_offset_penalty), weight=-100.0, params={})
  return cfg


register_mjlab_task(
  task_id=CARRY_V13_TASK_ID,
  env_cfg=luna_carry_steps_v13_env_cfg(),
  play_env_cfg=luna_carry_steps_v13_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


CARRY_V14_TASK_ID = "FG-Carry-Steps-Luna-V14"


class ArmScheduleAction(ArmLockedJointPositionAction):
  """V14：学習中、腕の目標を取り上げ・置く動作の範囲で動かし続ける（schedule=False なら外から arm_target を与える）。

  V13 の方策は、止まって腕を動かすと（そろえる・差し込む）、指令 0 でも胴体が 10 cm ほど後ろ・横へ動いた。
  学習では腕は決まった姿勢のままだったため（2026-10-12 昼、pick_place.py）
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.schedule = True
    self._bank = None

  def _init_bank(self):
    from fg.arm_ik import ArmIK

    ik = ArmIK()
    g = torch.Generator().manual_seed(0)

    def vec(**kw):
      r, err, _ = ik.solve(**kw)
      return torch.tensor([r[nm] for nm in self.arm_names]), err

    empty, carry = [], [vec()[0]]
    while len(empty) < 96:
      u = torch.rand(5, generator=g).tolist()
      # 取り上げ：上体から見た箱のずれ（前後 ±4 cm、左右 ±2.5 cm、高さ −2〜+7 cm、向き ±0.08 rad）、手の開き 0〜10 cm
      q, err = vec(dx=-0.04 + 0.08 * u[0], dy=-0.025 + 0.05 * u[1], dz=-0.02 + 0.09 * u[2], dyaw=-0.08 + 0.16 * u[3], spread=0.10 * u[4])
      if err < 0.005:
        empty.append(q)
    while len(carry) < 48:
      u = torch.rand(2, generator=g).tolist()
      # 置く：箱を下ろす（−7〜+4 cm）・前後 ±2 cm
      q, err = vec(dx=-0.02 + 0.04 * u[0], dz=-0.07 + 0.11 * u[1])
      if err < 0.005:
        carry.append(q)
    self._bank = (torch.stack(carry).to(self.device), torch.stack(empty).to(self.device))
    n = self.num_envs
    self._q_from = self._bank[0][0].expand(n, -1).clone()
    self._q_to = self._q_from.clone()
    self._k = torch.zeros(n, device=self.device)
    self._dur = torch.ones(n, device=self.device)
    self._next = torch.zeros(n, device=self.device)
    self.arm_target = self._q_from.clone()
    self.arm_target_mask = torch.ones(n, dtype=torch.bool, device=self.device)

  def _pick(self, ids, empty_mask):
    carry, empty = self._bank
    ic = torch.randint(0, len(carry), (len(ids),), device=self.device)
    ic = torch.where(torch.rand(len(ids), device=self.device) < 0.5, torch.zeros_like(ic), ic)  # 半分は抱える姿勢へ戻る
    ie = torch.randint(0, len(empty), (len(ids),), device=self.device)
    return torch.where(empty_mask[:, None], empty[ie], carry[ic])

  def reset_schedule(self, env_ids, empty_mask):
    """リセット：箱を持つ台は抱える姿勢、持たない台は取り上げの範囲のどれかから始める。"""
    if self._bank is None:
      self._init_bank()
    q = self._pick(env_ids, empty_mask)
    q = torch.where(empty_mask[:, None], q, self._bank[0][0])
    self._q_from[env_ids] = q
    self._q_to[env_ids] = q
    self.arm_target[env_ids] = q
    self._k[env_ids] = 0.0
    self._dur[env_ids] = 1.0
    self._next[env_ids] = 0.5 + 2.5 * torch.rand(len(env_ids), device=self.device)

  def process_actions(self, actions):
    if self.schedule and self._bank is not None:
      dt = self._env.step_dt
      self._k += dt
      ch = torch.nonzero(self._k >= self._next).flatten()
      if len(ch):
        self._q_from[ch] = self.arm_target[ch]
        self._q_to[ch] = self._pick(ch, is_empty(self._env)[ch])
        self._dur[ch] = 0.5 + 1.0 * torch.rand(len(ch), device=self.device)
        self._next[ch] = self._dur[ch] + 0.5 + 2.5 * torch.rand(len(ch), device=self.device)
        self._k[ch] = 0.0
      a = torch.clamp(self._k / self._dur, 0.0, 1.0)
      a = a * a * (3 - 2 * a)
      self.arm_target = self._q_from + (self._q_to - self._q_from) * a[:, None]
    super().process_actions(actions)


@dataclass(kw_only=True)
class ArmScheduleActionCfg(JointPositionActionCfg):
  def build(self, env):
    return ArmScheduleAction(self, env)


def randomize_carry_mode_v14(env, env_ids, fraction: float = EMPTY_FRACTION):
  """V13 の持つ／持たないの割り当て＋腕の動きの予定のリセット。"""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  empty = is_empty(env)
  e = torch.rand(len(env_ids), device=env.device) < fraction
  empty[env_ids] = e
  env.action_manager.get_term("joint_pos").reset_schedule(env_ids, e)
  ids = env_ids[e]
  if len(ids):
    box = env.scene["bin"]
    from fg.terrains import TILE_SIZE

    p = env.scene.env_origins[ids] + torch.tensor([TILE_SIZE / 2 - 0.4, TILE_SIZE / 2 - 0.4, 1.0], device=env.device)
    q = torch.tensor([1.0, 0, 0, 0], device=env.device).expand(len(ids), 4)
    box.write_root_link_pose_to_sim(torch.cat([p, q], dim=-1), env_ids=ids)
    box.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def luna_carry_steps_v14_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """V13 に「止まって・歩きながら腕を動かす（取り上げ・置く動作の範囲）」を足した版。止まる指令の台を 1 割 → 3 割。"""
  cfg = luna_carry_steps_v13_env_cfg(play=play)
  a = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = ArmScheduleActionCfg(**{f.name: getattr(a, f.name) for f in fields(a) if f.init})
  cfg.events["carry_mode"] = EventTermCfg(func=randomize_carry_mode_v14, mode="reset", params={})
  cfg.commands["twist"].rel_standing_envs = 0.3
  return cfg


register_mjlab_task(
  task_id=CARRY_V14_TASK_ID,
  env_cfg=luna_carry_steps_v14_env_cfg(),
  play_env_cfg=luna_carry_steps_v14_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


CARRY_V15_TASK_ID = "FG-Carry-Steps-Luna-V15"


def _standing(env) -> torch.Tensor:
  cmd = env.command_manager.get_command("twist")
  return (torch.linalg.norm(cmd[:, :2], dim=-1) < 0.05) & (cmd[:, 2].abs() < 0.05)


def stand_feet_still_penalty(env) -> torch.Tensor:
  """止まる指令のとき、足（足首）の速さの二乗の和 [m^2/s^2]。"""
  robot = env.scene["robot"]
  if not hasattr(env, "_fg_feet_ids"):
    env._fg_feet_ids = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"))[0]
  v = robot.data.body_link_lin_vel_w[:, env._fg_feet_ids]
  return torch.sum(torch.square(v), dim=(1, 2)) * _standing(env).float()


def stand_torso_still_penalty(env) -> torch.Tensor:
  """止まる指令のとき、上体（腕の付け根）の速さの二乗 [m^2/s^2]。上下は重く（受け具の高さが ±3 cm 揺れたため）。"""
  robot = env.scene["robot"]
  if not hasattr(env, "_fg_torso_id"):
    env._fg_torso_id = robot.find_bodies(("waist_pitch_link",))[0][0]
  v = robot.data.body_link_lin_vel_w[:, env._fg_torso_id]
  return (torch.sum(torch.square(v[:, :2]), dim=-1) + 4.0 * torch.square(v[:, 2])) * _standing(env).float()


def luna_carry_steps_v15_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """V14 に「止まる指令のときは、足を動かさず、上体を揺らさない」を足した版。

  run14 の方策は、止まる指令でも足踏みを続け、手を開いた構えで受け具の高さが 0.2 秒ごとに ±3 cm 揺れた。
  受け具とフランジの上下の許容は 3 mm なので、取り上げで差し込めなかった（2026-10-12 14 時、pick_place.py）
  """
  cfg = luna_carry_steps_v14_env_cfg(play=play)
  cfg.rewards["stand_feet_still"] = RewardTermCfg(func=stand_feet_still_penalty, weight=-5.0, params={})
  cfg.rewards["stand_torso_still"] = RewardTermCfg(func=stand_torso_still_penalty, weight=-50.0, params={})
  return cfg


register_mjlab_task(
  task_id=CARRY_V15_TASK_ID,
  env_cfg=luna_carry_steps_v15_env_cfg(),
  play_env_cfg=luna_carry_steps_v15_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
