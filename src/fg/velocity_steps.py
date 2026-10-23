"""Phase 1：2 段のステップの地形での速度追従（上り下り）。技術成果2・4（積載なし）に向けた最初の学習。

- 地形：src/fg/terrains.py の 2 段のステップ（段の高さ 0.05〜0.25 m をカリキュラムで上げる）＋平らな区画
- 入力：平坦路の 93 個に、胴体まわりの地形の高さ（mjlab の terrain_scan、1.6 × 1.0 m を 0.1 m 間隔）を足す。
  シミュレーターの正確な高さを使う（特権情報）。頭のカメラ＋地図への差し替えは Phase 2・3
- 報酬：平坦路と同じ。足の高さは mjlab 1.6.0 の TerrainHeightSensor で地面から測る
mjlab 同梱の G1 の凸凹地形の設定（tasks/velocity/config/g1/env_cfgs.py の rough）を手本にした。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from fg.luna_constants import LUNA_ACTION_SCALE, get_luna_robot_cfg
from fg.terrains import two_step_terrain_generator_cfg
from fg.velocity_flat import FOOT_GEOM_NAMES, ROOT_BODY, SITE_NAMES, TORSO_BODY, luna_flat_env_cfg, luna_ppo_runner_cfg

STEPS_TASK_ID = "FG-Velocity-Steps-Luna"


def luna_steps_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # 段差の接触の設定（G1 の凸凹地形を手本に、計算の重さを見て調整）
  cfg.sim.mujoco.ccd_iterations = 50  # G1 の凸凹地形は 500。Luna のメッシュの当たり判定では 500 だと 1 反復 約 13 秒かかったので 50（約 1.3 秒）
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_luna_robot_cfg()}

  # 地形：2 段のステップ＋平らな区画。行ごとに段が高くなる。最初は下から 4 行目までに置く
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = two_step_terrain_generator_cfg(num_rows=10, num_cols=20)
  cfg.scene.terrain.max_init_terrain_level = 3

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = ROOT_BODY
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LUNA_ACTION_SCALE
  cfg.viewer.body_name = TORSO_BODY

  # 速度指令：段差ではゆっくり〜普通の速さに絞る（前後 −0.5〜1.0 m/s）。速度を上げるカリキュラムは外す
  tw = cfg.commands["twist"]
  assert isinstance(tw, UniformVelocityCommandCfg)
  tw.viz.z_offset = 1.3
  tw.ranges.lin_vel_x = (-0.5, 1.0)
  tw.ranges.lin_vel_y = (-0.3, 0.3)
  tw.ranges.ang_vel_z = (-0.7, 0.7)
  cfg.curriculum.pop("command_vel", None)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # 姿勢・上体・足の報酬の設定は平坦路の版と同じ値を使う
  flat = luna_flat_env_cfg()
  for k in ("std_standing", "std_walking", "std_running"):
    cfg.rewards["pose"].params[k] = flat.rewards["pose"].params[k]
  cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO_BODY,)
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})
    tg = cfg.scene.terrain.terrain_generator
    tg.curriculum = False
    tg.num_cols = 5
    tg.num_rows = 5
    tg.border_width = 10.0

  return cfg


def luna_steps_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = luna_ppo_runner_cfg()
  cfg.experiment_name = "luna_velocity_steps"
  cfg.max_iterations = 4000
  return cfg


NOSTALL_TASK_ID = "FG-Velocity-Steps-Luna-NoStall"


def stall_penalty(env, command_name: str, min_command: float = 0.1, min_speed: float = 0.1):
  """立ち止まりへの罰（作者の決定、2026-10-11「立ち止まりは転倒同様に失敗なんだから、
  速度 0.1 未満はペナルティにしよう」）。

  歩けという指令（前後・横の大きさが min_command 以上）なのに、指令の向きへの速度が min_speed 未満なら 1。
  速さだけでなく「指令の向きへの成分」で測るのは、run5b が 20 cm の段の前で向きを変えて横へ歩き、
  段を避けていたため（横へ歩いても前へは進んでいない）。
  """
  import torch

  robot = env.scene["robot"]
  cmd = env.command_manager.get_command(command_name)[:, :2]
  cmd_norm = torch.norm(cmd, dim=1)
  direction = cmd / cmd_norm.clamp(min=1e-6).unsqueeze(1)
  speed_along = torch.sum(robot.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
  return ((cmd_norm >= min_command) & (speed_along < min_speed)).float()


def luna_steps_nostall_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """run5b に「立ち止まりの罰」だけを足した版（他は同じ。変えたのは 1 か所）。"""
  from mjlab.managers.reward_manager import RewardTermCfg

  cfg = luna_steps_env_cfg(play=play)
  # 重みは速度追従のご褒美の最大（2.0）と同じ大きさにした：止まると、追従できたときのご褒美を丸ごと失うのと同じ
  cfg.rewards["stall"] = RewardTermCfg(func=stall_penalty, weight=-2.0, params={"command_name": "twist"})
  return cfg


V7_TASK_ID = "FG-Velocity-Steps-Luna-V7"


def _yaw_left_axis(robot):
  import torch

  q = robot.data.root_link_quat_w  # (w, x, y, z)
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  return -torch.sin(yaw), torch.cos(yaw)  # 胴体の向きで見た「左」方向（水平）


def feet_cross_penalty(env, min_sep: float = 0.15):
  """足の交差・内股への罰。胴体の向きで見た左右の足の横の間隔（左足 − 右足）が min_sep より狭い分 [m]。

  run6 の 20 cm 上りの失敗を 256 台で分析したところ、転倒した 15 台はすべて、左右の足が交差していた
  （横の間隔の最小が平均 −21 cm）。Luna の脚どうしの衝突を切っているので、実機ではありえない足運びを覚えた。
  """
  import torch

  robot = env.scene["robot"]
  ids, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
  fp = robot.data.site_pos_w[:, ids, :2] - robot.data.root_link_pos_w[:, None, :2]
  lx, ly = _yaw_left_axis(robot)
  lat = lx[:, None] * fp[..., 0] + ly[:, None] * fp[..., 1]
  sep = lat[:, 0] - lat[:, 1]
  return torch.clamp(min_sep - sep, min=0.0)


def feet_stumble_penalty(env, sensor_name: str = "feet_ground_contact", ratio: float = 2.0):
  """つまずきへの罰：足に横向きの力が、縦の力の ratio 倍より大きくかかったら 1（つま先が段の縦の面にぶつかる等）。"""
  import torch

  data = env.scene[sensor_name].data
  f = data.force  # [B, 2, 3]、世界座標（netforce）
  horiz = torch.norm(f[..., :2], dim=-1)
  vert = torch.abs(f[..., 2])
  return torch.any(horiz > ratio * vert + 1.0, dim=1).float()


def track_linear_velocity_xy(env, std: float, command_name: str):
  """速度追従のご褒美（水平だけ）。mjlab の既定は胴体の上下の速度も誤差に含めるので、段を上ること自体が損になる。"""
  import torch

  robot = env.scene["robot"]
  cmd = env.command_manager.get_command(command_name)
  err = torch.sum(torch.square(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-err / std**2)


def luna_steps_v7_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """run6 に、20 cm の上りを確実にするための変更を足した版（作者 の合格基準：上りで転倒 0・立ち止まり 0）。
  1. 足の交差・内股への罰（重み −20、横の間隔 15 cm 未満の分）
  2. つまずき（足への横向きの力）への罰（重み −1）
  3. 速度追従のご褒美から胴体の上下の速度を外す
  4. 遊脚の高さの目標を 0.10 → 0.12 m（地面からの高さ）
  """
  from mjlab.managers.reward_manager import RewardTermCfg

  cfg = luna_steps_nostall_env_cfg(play=play)
  cfg.rewards["feet_cross"] = RewardTermCfg(func=feet_cross_penalty, weight=-20.0, params={"min_sep": 0.15})
  cfg.rewards["feet_stumble"] = RewardTermCfg(func=feet_stumble_penalty, weight=-1.0, params={})
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_linear_velocity_xy, weight=2.0, params={"command_name": "twist", "std": 0.5}
  )
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
  return cfg


V8_TASK_ID = "FG-Velocity-Steps-Luna-V8"


def luna_steps_v8_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """run7 に足した版（run7 の途中評価を受けて）。
  1. 段の高さを 10〜27 cm に寄せる（20 cm の前後で多く練習させる）
  2. 足の間隔の下限を 15 → 18 cm（run7 の途中で、転倒した台は足の間隔がほぼ 0 だった）
  3. 前後の速度指令を −0.3〜0.8 m/s（後ろ向きを減らし、ゆっくり〜人の階段の速さを多く）
  """
  from fg.terrains import two_step_hard_terrain_generator_cfg

  cfg = luna_steps_v7_env_cfg(play=play)
  cfg.rewards["feet_cross"].params["min_sep"] = 0.18
  tw = cfg.commands["twist"]
  tw.ranges.lin_vel_x = (-0.3, 0.8)
  if not play:
    cfg.scene.terrain.terrain_generator = two_step_hard_terrain_generator_cfg(num_rows=10, num_cols=20)
  return cfg


V9_TASK_ID = "FG-Velocity-Steps-Luna-V9"


def slow_progress_penalty(env, command_name: str, min_command: float = 0.1, ratio: float = 0.5, floor: float = 0.1):
  """立ち止まりの罰を「指令の半分未満の速さ」に広げたもの。

  run7 の最終を 256 台で調べたところ、時間切れの台の多くは段の前で止まるのではなく、
  1 段目に乗ってから段に沿って斜め・横へ歩き去っていた（向きのずれ平均 73〜81 度）。
  「指令の向きへ 0.1 m/s」の罰は、斜めにゆっくり進めば避けられてしまう。
  """
  import torch

  robot = env.scene["robot"]
  cmd = env.command_manager.get_command(command_name)[:, :2]
  cmd_norm = torch.norm(cmd, dim=1)
  direction = cmd / cmd_norm.clamp(min=1e-6).unsqueeze(1)
  speed_along = torch.sum(robot.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
  need = torch.clamp(ratio * cmd_norm, min=floor)
  return ((cmd_norm >= min_command) & (speed_along < need)).float()


def lin_vel_error_penalty(env, command_name: str):
  """水平の速度の指令との誤差の二乗 [(m/s)^2]。追従のご褒美（std 0.5 で緩い）だけでは、横へのずれが残った。"""
  import torch

  robot = env.scene["robot"]
  cmd = env.command_manager.get_command(command_name)
  return torch.sum(torch.square(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]), dim=1)


def luna_steps_v9_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """run8 に足した版（run7 の最終の失敗の分析を受けて、2026-10-12 4 時ごろ）。
  1. 立ち止まりの罰：指令の向きへの速さが「指令の半分（最低 0.1 m/s）」未満なら罰（重み −2 のまま）
  2. 速度の誤差の二乗への罰（重み −1）：段に沿って横・斜めへ逃げる動きを減らす
  """
  from mjlab.managers.reward_manager import RewardTermCfg

  cfg = luna_steps_v8_env_cfg(play=play)
  cfg.rewards["stall"] = RewardTermCfg(func=slow_progress_penalty, weight=-2.0, params={"command_name": "twist"})
  cfg.rewards["lin_vel_error"] = RewardTermCfg(func=lin_vel_error_penalty, weight=-1.0, params={"command_name": "twist"})
  return cfg


V10_TASK_ID = "FG-Velocity-Steps-Luna-V10"


def ang_vel_error_penalty(env, command_name: str):
  """旋回速度の指令との誤差の二乗 [(rad/s)^2]。

  run9 を 1024 台で評価すると、残った失敗（ゆっくりの上り・25 cm の段）は、旋回の指令 0 なのに
  向きを 60〜120 度変えて段から離れる時間切れだった（向きを保つ仕組みを足すと 0 になった）。
  """
  import torch

  robot = env.scene["robot"]
  cmd = env.command_manager.get_command(command_name)
  return torch.square(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])


def luna_steps_v10_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """run9 に、旋回速度の誤差の二乗への罰（重み −0.5）だけを足した版。"""
  from mjlab.managers.reward_manager import RewardTermCfg

  cfg = luna_steps_v9_env_cfg(play=play)
  cfg.rewards["ang_vel_error"] = RewardTermCfg(func=ang_vel_error_penalty, weight=-0.5, params={"command_name": "twist"})
  return cfg


register_mjlab_task(
  task_id=V10_TASK_ID,
  env_cfg=luna_steps_v10_env_cfg(),
  play_env_cfg=luna_steps_v10_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=V9_TASK_ID,
  env_cfg=luna_steps_v9_env_cfg(),
  play_env_cfg=luna_steps_v9_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=V8_TASK_ID,
  env_cfg=luna_steps_v8_env_cfg(),
  play_env_cfg=luna_steps_v8_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=V7_TASK_ID,
  env_cfg=luna_steps_v7_env_cfg(),
  play_env_cfg=luna_steps_v7_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),  # 実験名は run5b・run6 と同じ（続きから再開するため）
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=NOSTALL_TASK_ID,
  env_cfg=luna_steps_nostall_env_cfg(),
  play_env_cfg=luna_steps_nostall_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),  # 実験名は run5b と同じ（続きから学習を再開するため）
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=STEPS_TASK_ID,
  env_cfg=luna_steps_env_cfg(),
  play_env_cfg=luna_steps_env_cfg(play=True),
  rl_cfg=luna_steps_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
