"""Luna の平坦路での速度追従（歩行）タスク。mjlab 1.6.0 用。

mjlab 同梱の G1 の設定（mjlab/tasks/velocity/config/g1/env_cfgs.py）を手本に、
体の部位の名前を Luna に合わせ、まずは「ゆっくり前へ歩く」に絞った速度指令にしている。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from fg.luna_constants import LUNA_ACTION_SCALE, get_luna_robot_cfg

TASK_ID = "FG-Velocity-Flat-Luna"

ROOT_BODY = "base_link"  # G1 の pelvis に相当
TORSO_BODY = "waist_pitch_link"  # G1 の torso_link に相当（腰より上の上体）
SITE_NAMES = ("left_foot", "right_foot")
FOOT_GEOM_NAMES = tuple(f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(9))


def luna_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_luna_robot_cfg()}

  # 平坦路にする（地形のスキャンも外す）
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)

  # mjlab 1.6.0：足の高さは地形からの高さで測る（TerrainHeightSensor）。足裏の site から真下に光線を出す
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
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

  # まずは「ゆっくり歩く」：前後 −0.3〜0.6 m/s、横 ±0.2 m/s、旋回 ±0.5 rad/s。速度を上げるカリキュラムは外す
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.3
  twist_cmd.ranges.lin_vel_x = (-0.3, 0.6)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  cfg.curriculum.pop("command_vel", None)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  # 姿勢の報酬の許容幅（G1 の値を Luna の関節名に合わせたもの。Luna は手首が yaw のみ）
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*head.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = dict(cfg.rewards["pose"].params["std_walking"])

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
    cfg.curriculum = {}
    cfg.terminations.pop("out_of_terrain_bounds", None)

  return cfg


def luna_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="luna_velocity_flat",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
  )


GAIT_TASK_ID = "FG-Velocity-Flat-Luna-Gait"


def luna_flat_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """run1 は足をほとんど上げずに摺り足で歩いた（足の最高点の平均 約 9 mm）。
  段差を上るには足を上げる歩き方が要るので、足を上げる方向の報酬を強めた版。"""
  cfg = luna_flat_env_cfg(play=play)
  # 片足ずつ 0.05〜0.5 秒浮かせることへのご褒美（G1 では 0 にしてある）。歩く指令（0.1 以上）のときだけ
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params["command_threshold"] = 0.1
  # 遊脚の高さが目標に届かないことへの罰を強める（−0.25 → −1.0）、目標は 8 cm
  cfg.rewards["foot_swing_height"].weight = -1.0
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  return cfg


def luna_gait_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = luna_ppo_runner_cfg()
  cfg.experiment_name = "luna_velocity_flat_gait"
  return cfg


FAST_TASK_ID = "FG-Velocity-Flat-Luna-Fast"


def luna_flat_fast_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """作者 の仮説（2026-10-11）：「速い速度指令を出せば、自然に前へ進むフォームができて、
  足も摺り足でなく上がってくるのでは」。それを確かめるため、報酬は run1 と同じまま、
  速度指令の範囲だけを広げた版（run1 との差は速度指令だけ）。"""
  cfg = luna_flat_env_cfg(play=play)
  tw = cfg.commands["twist"]
  assert isinstance(tw, UniformVelocityCommandCfg)
  tw.ranges.lin_vel_x = (-0.5, 1.5)
  tw.ranges.lin_vel_y = (-0.3, 0.3)
  tw.ranges.ang_vel_z = (-0.7, 0.7)
  return cfg


def luna_fast_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = luna_ppo_runner_cfg()
  cfg.experiment_name = "luna_velocity_flat_fast"
  return cfg


CURRICULUM_TASK_ID = "FG-Velocity-Flat-Luna-Curriculum"

# 1 反復 = 1 環境あたり 24 ステップ（num_steps_per_env）。カリキュラムの切り替えは環境ステップで指定する
_STEPS_PER_ITER = 24


def luna_flat_curriculum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """前進 2.0 m/s まで、速度の範囲を段階的に広げる版（作者の決定、2026-10-11）。
  報酬は run1 と同じ。段階の切り替えは反復数で決める（mjlab の commands_vel は成績で判定しない）。"""
  cfg = luna_flat_env_cfg(play=play)
  tw = cfg.commands["twist"]
  assert isinstance(tw, UniformVelocityCommandCfg)
  # Stage 1 の範囲から始める
  tw.ranges.lin_vel_x = (-0.5, 0.5)
  tw.ranges.lin_vel_y = (-0.5, 0.5)
  tw.ranges.ang_vel_z = (-1.0, 1.0)
  if not play:
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 0.5)},  # Stage 1：低速の安定
          {"step": 600 * _STEPS_PER_ITER, "lin_vel_x": (-0.5, 1.0)},  # Stage 2：通常歩行
          {"step": 1200 * _STEPS_PER_ITER, "lin_vel_x": (-0.5, 1.5)},  # Stage 3：速度追従
          {"step": 1800 * _STEPS_PER_ITER, "lin_vel_x": (-0.5, 2.0)},  # Stage 4：2.0 m/s
        ],
      },
    )
  else:
    tw.ranges.lin_vel_x = (-0.5, 2.0)
  return cfg


def luna_curriculum_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = luna_ppo_runner_cfg()
  cfg.experiment_name = "luna_velocity_flat_curriculum"
  cfg.max_iterations = 3000
  return cfg


register_mjlab_task(
  task_id=CURRICULUM_TASK_ID,
  env_cfg=luna_flat_curriculum_env_cfg(),
  play_env_cfg=luna_flat_curriculum_env_cfg(play=True),
  rl_cfg=luna_curriculum_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=FAST_TASK_ID,
  env_cfg=luna_flat_fast_env_cfg(),
  play_env_cfg=luna_flat_fast_env_cfg(play=True),
  rl_cfg=luna_fast_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=GAIT_TASK_ID,
  env_cfg=luna_flat_gait_env_cfg(),
  play_env_cfg=luna_flat_gait_env_cfg(play=True),
  rl_cfg=luna_gait_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=luna_flat_env_cfg(),
  play_env_cfg=luna_flat_env_cfg(play=True),
  rl_cfg=luna_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
