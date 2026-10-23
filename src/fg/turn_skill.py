"""回れ右（その場で向きを変える）だけの方策の課題（2026-10-17）。

統合（integrate.py）で、置いた後のその場での振り返りに 1 往復 21 秒かかっていた（うち 13 秒はほぼ止まっていた）。
run14 は「回れ」の指令 0.5 rad/s に対して、その場では 0.05〜0.1 rad/s しか回らなかった（前へ 0.12 m/s 歩きながらなら 0.5〜0.7 rad/s 回る）。
run14 の学習では、前後・横の指令が 0 で回る指令だけの台が少なかったため。

課題 V14（腕を動かし続ける・箱を持つ／持たない）を元に、指令を「その場で回る」に絞る：
- 前後・横の速さの指令は 0。回る速さは ±1.2 rad/s。7 割の台は向きの指令（目標の向きとの差 × 1.0 を回る速さに。±180 度）
- 回る速さに従う報酬を重く（track_angular_velocity 2 → 4、ang_vel_error −0.3 → −1.0）
入力・行動は run14 と同じ（run14 から始めて、統合では振り返るときだけこの方策に替える）。
"""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from fg.carry_steps import luna_carry_steps_v14_env_cfg
from fg.velocity_steps import luna_steps_ppo_runner_cfg

TURN_TASK_ID = "FG-Turn-Luna-V1"


def luna_turn_env_cfg(play: bool = False):
  cfg = luna_carry_steps_v14_env_cfg(play=play)
  tw = cfg.commands["twist"]
  tw.ranges.lin_vel_x = (0.0, 0.0)
  tw.ranges.lin_vel_y = (0.0, 0.0)
  tw.ranges.ang_vel_z = (-1.2, 1.2)
  tw.heading_command = True
  tw.rel_heading_envs = 0.7
  tw.heading_control_stiffness = 1.0
  tw.rel_forward_envs = 0.0
  tw.rel_standing_envs = 0.15
  cfg.rewards["track_angular_velocity"].weight = 4.0
  cfg.rewards["ang_vel_error"].weight = -1.0
  return cfg


register_mjlab_task(
  task_id=TURN_TASK_ID,
  env_cfg=luna_turn_env_cfg(),
  play_env_cfg=luna_turn_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


TURN_V2_TASK_ID = "FG-Turn-Luna-V2"


def turn_progress_penalty(env, command_name: str = "twist", min_command: float = 0.15):
  """回る指令に対して、指令の向きへの回る速さが指令の半分（最低 0.08 rad/s）に届かない分の二乗（0〜1）。

  小さい指令（0.3 rad/s 付近）でほとんど回らない（統合で残り 15 度から 2 秒かかった）のを減らす（2026-10-17）
  """
  import torch

  c = env.command_manager.get_command(command_name)[:, 2]
  v = torch.sign(c) * env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  q = torch.clamp(0.5 * c.abs(), min=0.08)
  return torch.square(torch.clamp((q - v) / q, 0.0, 1.0)) * (c.abs() >= min_command).float()


def hold_margin_penalty(env):
  """箱を持つ台の、受け具からのずれ d が 15 mm を超えた分（40 mm で 1）の二乗。落とす（40 mm）前に余裕を残す"""
  import torch

  from fg.carry_steps import is_empty
  from fg.free_bin import bin_offset_from_hand

  d = bin_offset_from_hand(env)
  return torch.square(torch.clamp((d - 0.015) / 0.025, 0.0, 1.0)) * (~is_empty(env)).float()


def luna_turn_v2_env_cfg(play: bool = False):
  """V1 に「箱を持ったまま速く回っても落とさない」を足した版（2026-10-17）。

  統合で台の上の振り返り（箱を持つ）にも V1 を使い、最小 0.6 rad/s にすると、箱を落とす台が 5 台出た（run14 では 0）。
  - 箱を持つ台 9 割（V1 は 7 割）。腕は動かさない（統合の振り返りの間、腕は決まった姿勢）
  - 回る速さに従う報酬の幅を狭く（std 0.71 → 0.35。指令 0.3 で止まっていても満点の 84 % 取れていた）、回る進みの不足の罰 −1.5
  - 箱：落とす −500 → −1000、傾き −2 → −6、ずれ −100 → −300、ずれの余裕の不足 −2
  - 回る速さ ±1.0 rad/s（統合の上限に合わせる）、向きの指令 6 割（剛性 2.5）
  """
  from dataclasses import fields

  import fg.carry_steps as cs
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.reward_manager import RewardTermCfg

  cfg = luna_turn_env_cfg(play=play)
  a = cfg.actions["joint_pos"]

  @__import__("dataclasses").dataclass(kw_only=True)
  class _FixedArmCfg(cs.ArmScheduleActionCfg):
    def build(self, env):
      act = cs.ArmScheduleAction(self, env)
      act.schedule = False  # リセットのときの姿勢（箱を持つ台は抱える姿勢）のまま
      return act

  cfg.actions["joint_pos"] = _FixedArmCfg(**{f.name: getattr(a, f.name) for f in fields(a) if f.init})
  cfg.events["carry_mode"] = EventTermCfg(func=cs.randomize_carry_mode_v14, mode="reset", params={"fraction": 0.1})
  tw = cfg.commands["twist"]
  tw.ranges.ang_vel_z = (-1.0, 1.0)
  tw.rel_heading_envs = 0.6
  tw.heading_control_stiffness = 2.5
  tw.rel_standing_envs = 0.1
  cfg.rewards["track_angular_velocity"].params["std"] = 0.35
  cfg.rewards["turn_progress"] = RewardTermCfg(func=turn_progress_penalty, weight=-1.5, params={"command_name": "twist"})
  cfg.rewards["hold_margin"] = RewardTermCfg(func=hold_margin_penalty, weight=-2.0, params={})
  cfg.rewards["bin_drop"].weight = -1000.0
  cfg.rewards["bin_tilt"].weight = -6.0
  cfg.rewards["bin_offset"].weight = -300.0
  return cfg


register_mjlab_task(
  task_id=TURN_V2_TASK_ID,
  env_cfg=luna_turn_v2_env_cfg(),
  play_env_cfg=luna_turn_v2_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
