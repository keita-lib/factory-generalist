"""LimX Luna（HU_L04）を mjlab で使うためのロボット設定。

mjlab 同梱の Unitree G1 の設定（mjlab/asset_zoo/robots/unitree_g1/g1_constants.py）を手本にしている。

モデルは luna-description の運動学用 MJCF（HU_L04_01_vis.xml）を使う。
足首と腰の平行リンクを持たない「直列近似」のモデルで、足首 pitch/roll・腰 roll/pitch を
関節として直接駆動する（URDF と同じ近似。G1 の mjlab 設定も同じ考え方）。
平行リンク付きのモデル（HU_L04_01.xml）での学習は、後で比較する。
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF とアセット
##

REPO_ROOT = Path(__file__).resolve().parents[2]
LUNA_DIR = REPO_ROOT / "third_party" / "luna-description" / "HU_L04_description"
LUNA_XML = LUNA_DIR / "xml" / "HU_L04_01_vis.xml"
LUNA_MESH_DIR = LUNA_DIR / "meshes" / "HU_L04_01"
assert LUNA_XML.exists(), LUNA_XML

# 足の当たり判定（箱 1 個＋球 8 個）の、足裏の最下点の高さ（ankle_roll_link 座標、m）
# 球の中心 z=-0.06846、半径 0.005 → 最下点 -0.07346
FOOT_SOLE_Z = -0.06846 - 0.005
FOOT_SITE_POS = (0.018, 0.0, FOOT_SOLE_Z)  # 箱の中心の真下


def _edit_spec(spec: mujoco.MjSpec) -> None:
  """mjlab のタスクが前提にしている名前・センサーを Luna のモデルに足す。"""
  # 床・照明・カメラは mjlab の地形側が持つので、ロボットのモデルからは外す
  for g in list(spec.worldbody.geoms):
    spec.delete(g)
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  for cam in list(spec.worldbody.cameras):
    spec.delete(cam)

  base = spec.body("base_link")
  base.joints[0].name = "floating_base_joint"

  # 当たり判定の geom に名前を付ける（G1 の "*_collision" の命名に合わせる）。
  # グループも G1 に合わせる（見た目 2、当たり判定 3）。Luna の MJCF は当たり判定が group 0 で、
  # mjlab の地形スキャン（group 0 だけに当たる＝「地形だけ」の前提）が Luna 自身の脚に当たってしまうため
  for body in spec.bodies:
    n = 0
    for g in body.geoms:
      if g.contype == 0 and g.conaffinity == 0:
        g.group = 2  # 見た目用
        continue
      g.group = 3
      if body.name.endswith("ankle_roll_link"):
        side = "left" if body.name.startswith("left") else "right"
        g.name = f"{side}_foot{n}_collision"
      else:
        g.name = f"{body.name}_collision{n}"
      n += 1

  # 足裏の site（足の高さ・遊脚の高さの報酬に使う）
  for side in ("left", "right"):
    spec.body(f"{side}_ankle_roll_link").add_site(
      name=f"{side}_foot", pos=FOOT_SITE_POS, size=[0.01, 0.01, 0.01]
    )

  # IMU とルートの角運動量のセンサー（速度追従タスクの観測・報酬が名前で参照する）
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="base_imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="base_imu",
  )
  spec.add_sensor(
    name="imu_lin_acc",
    type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="base_imu",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="base_link",
  )


def get_spec() -> mujoco.MjSpec:
  # mjlab 1.6.0 ではメッシュを assets に詰めずに、MJCF のパスのまま読み込む（G1 の設定と同じ）
  spec = mujoco.MjSpec.from_file(str(LUNA_XML))
  _edit_spec(spec)
  return spec


##
# アクチュエータ
##

# SRDF（HU_L04_01.srdf）のロータ慣性 [kg m^2] と減速比から、関節側の反射慣性 I×N^2 を求める
def _armature(rotor_inertia: float, gear: float) -> float:
  return rotor_inertia * gear**2


ARMATURE_HIP_KNEE = _armature(0.000163107, 21.1765)  # 股 3 軸・膝
ARMATURE_MID = _armature(3.88572e-05, 21.391)  # 足首・腰・肩
ARMATURE_ELBOW = _armature(2.79006e-05, 21.391)  # 肘
ARMATURE_SMALL = _armature(2.06106e-05, 28.1716)  # 頭・手首

# G1 と同じ決め方：固有振動数 10 Hz、減衰比 2
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _gains(armature: float) -> tuple[float, float]:
  return armature * NATURAL_FREQ**2, 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _actuator(names: tuple[str, ...], armature: float, effort: float) -> BuiltinPositionActuatorCfg:
  kp, kd = _gains(armature)
  return BuiltinPositionActuatorCfg(
    target_names_expr=names,
    stiffness=kp,
    damping=kd,
    effort_limit=effort,
    armature=armature,
  )


# トルク上限は URDF の effort（股・膝 200、足首・腰・肩 70、肘 30、頭・手首 15.8 Nm）
LUNA_ACTUATOR_LEG = _actuator(
  (".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"),
  ARMATURE_HIP_KNEE,
  200.0,
)
# 足首と腰の roll/pitch は実機では 2 個のモーターの平行リンク。G1 の設定と同じく、
# 公称姿勢での等価な反射慣性を 2 個分の和で近似する
LUNA_ACTUATOR_ANKLE = _actuator(
  (".*_ankle_pitch_joint", ".*_ankle_roll_joint"), ARMATURE_MID * 2, 70.0
)
LUNA_ACTUATOR_WAIST_RP = _actuator(
  ("waist_roll_joint", "waist_pitch_joint"), ARMATURE_MID * 2, 70.0
)
LUNA_ACTUATOR_WAIST_YAW = _actuator(("waist_yaw_joint",), ARMATURE_MID, 70.0)
LUNA_ACTUATOR_SHOULDER = _actuator(
  (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"),
  ARMATURE_MID,
  70.0,
)
LUNA_ACTUATOR_ELBOW = _actuator((".*_elbow_joint",), ARMATURE_ELBOW, 30.0)
LUNA_ACTUATOR_SMALL = _actuator(
  ("head_yaw_joint", "head_pitch_joint", ".*_wrist_yaw_joint"), ARMATURE_SMALL, 15.8
)

LUNA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    LUNA_ACTUATOR_LEG,
    LUNA_ACTUATOR_ANKLE,
    LUNA_ACTUATOR_WAIST_RP,
    LUNA_ACTUATOR_WAIST_YAW,
    LUNA_ACTUATOR_SHOULDER,
    LUNA_ACTUATOR_ELBOW,
    LUNA_ACTUATOR_SMALL,
  ),
  soft_joint_pos_limit_factor=0.9,
)

##
# 初期姿勢（膝を少し曲げた立ち姿勢）。check_keyframe.py で確認：足裏はほぼ水平（つま先が約2°下がり）
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.81),  # 足裏が床に付く高さ 0.803 m（check_keyframe.py）＋余裕
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.25,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    ".*_elbow_joint": -0.3,
  },
  joint_vel={".*": 0.0},
)

##
# 当たり判定
##

FOOT_GEOM_EXPR = r"^(left|right)_foot[0-8]_collision$"

# ロボット同士（自己）の衝突は切り、地面との接触だけを残す（まず歩けることを優先）。
# 足は condim=3（摩擦あり）、それ以外は condim=1
COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  contype=0,
  conaffinity=1,
  condim={FOOT_GEOM_EXPR: 3, ".*_collision.*": 1},
  priority={FOOT_GEOM_EXPR: 1, ".*": 0},  # mjlab 1.6.0 から全 geom に明示が必要
  friction={FOOT_GEOM_EXPR: (0.6,)},
)


def get_luna_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(COLLISION_WITHOUT_SELF,),
    spec_fn=get_spec,
    articulation=LUNA_ARTICULATION,
  )


LUNA_ACTION_SCALE: dict[str, float] = {}
for a in LUNA_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  assert a.effort_limit is not None
  for n in a.target_names_expr:
    LUNA_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness


##
# 回収箱を両手で持つ（積載の評価用）
##

# 箱を前に抱える腕の姿勢（src/fg/carry_pose_ik.py で求めた。手首が胴体の前 0.30 m・床から 1.02 m・左右 ±0.183 m）
# ±0.183 m は、TRUSCO THC-04B の口のフランジの下に手の甲が入る幅（2026-10-12 に ±0.20 m から変更。±0.177 m では手が胴の壁に 4.9 mm めり込んだので ±0.183 m）
CARRY_ARM_POSE: dict[str, float] = {
  "left_shoulder_pitch_joint": -0.464, "left_shoulder_roll_joint": 0.013,
  "left_shoulder_yaw_joint": -0.079, "left_elbow_joint": -0.893,
  "right_shoulder_pitch_joint": -0.464, "right_shoulder_roll_joint": -0.013,
  "right_shoulder_yaw_joint": 0.079, "right_elbow_joint": -0.893,
}

# 回収箱：TRUSCO THC 型コンテナ THC-04B（青、蓋なし）を模す（作者 の指定、2026-10-12）
# 仕様（ミスミの THC-04B-OD の掲載値。色違いで寸法は同じと仮定）：外形 298 × 207 × 100 mm、有効内寸 246 × 165 × 96 mm、
# 質量 0.343 kg、容量 4.3 L。口のまわりに取っ手（フランジ）があり、手の甲に引っかかって滑り落ちにくい（作者）
# モデル：長い辺（298）を左右（両手の間）に向ける。胴（壁は鉛直、実物のテーパーは省略）＋口のまわりの水平なフランジ。
# 手は胴の壁のすぐ外にあり、フランジ（胴から外へ 22 mm）が手の甲の上にかぶさる。
BIN_OUTER = (0.207, 0.298, 0.100)  # 前後・左右・高さ [m]（フランジを含む外形）
BIN_INNER = (0.165, 0.246, 0.096)  # 有効内寸 [m]
BIN_WALL = 0.004  # 壁・底の厚さ [m]（仮）
BIN_FLANGE_T = 0.005  # フランジの厚さ [m]（仮）
BIN_CASE_MASS = 0.343  # [kg]
CAN_RADIUS = 0.033  # 350 mL 缶の半径 [m]（直径 約 66 mm）
CAN_HALF_HEIGHT = 0.061  # 350 mL 缶の高さの半分 [m]（高さ 約 122 mm。内寸の深さ 96 mm より 26 mm 高く、口から出る）
CAN_MASS = 0.37  # 中身 約 355 g ＋ 缶 約 15 g [kg]（仮）
BIN_BLUE = [0.12, 0.38, 0.85, 1.0]
CAN_RGBA = [0.85, 0.75, 0.35, 1.0]


def _pose_qpos(m: mujoco.MjModel) -> "np.ndarray":
  import re

  q = m.qpos0.copy()
  q[2] = KNEES_BENT_KEYFRAME.pos[2]
  q[3:7] = [1, 0, 0, 0]
  for j in range(m.njnt):
    name = m.joint(j).name
    for pat, v in {**KNEES_BENT_KEYFRAME.joint_pos, **CARRY_ARM_POSE}.items():
      if re.fullmatch(pat, name):
        q[m.jnt_qposadr[j]] = v
  return q


def _hand_points(m: mujoco.MjModel, d: mujoco.MjData, side: str):
  """抱える姿勢での手（手首 yaw リンクの当たり判定メッシュ）の頂点（世界座標）。"""
  import numpy as np

  b = m.body(f"{side}_wrist_yaw_link").id
  pts = []
  for g in range(m.ngeom):
    if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] + m.geom_conaffinity[g] > 0:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      pts.append((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g])
  return np.concatenate(pts)


def _add_box(body, name: str, half, pos, mass: float, rgba, contype: int = 0, conaffinity: int = 1, condim: int = 1,
             friction=None) -> None:
  # 当たり判定（既定は地面とだけ当たる、group 3）と見た目（group 2）を同じ形で 1 組
  extra = {} if friction is None else {"friction": list(friction)}
  body.add_geom(name=f"{name}_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos),
                mass=mass, rgba=rgba, group=3, contype=contype, conaffinity=conaffinity, condim=condim, **extra)
  body.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos),
                density=0, rgba=rgba, group=2, contype=0, conaffinity=0)


def add_reject_bin(spec: mujoco.MjSpec, num_cans: int = 3) -> None:
  """THC-04B を模した回収箱を左手首の子として付け、右手首とは weld 拘束でつなぐ（両手で持つ閉ループ）。
  中に 350 mL 缶を num_cans 本、立てて並べる。

  置き方：フランジの下面を、抱える姿勢での手の甲（フランジの真下に入る部分）のいちばん高いところに合わせる
  （フランジが手の甲に乗る）。
  前後は手の中ほど。左右は両手の真ん中。
  簡略化：缶はケースに固定（中で転がらない）。手の甲とフランジは接触で支えるのではなく、手首への固定と weld で支える
  （見た目と力のかかる位置を合わせた形）。weld の相対姿勢は抱える姿勢で計算して明示する（既定だと関節角 0 の姿勢になるため）。
  """
  import numpy as np

  L = bin_layout(spec)
  d, rw, center_w, pos_l, q_l = L["data"], L["rw"], L["center_w"], L["pos_l"], L["q_l"]
  b = spec.body("left_wrist_yaw_link").add_body(name="reject_bin", pos=pos_l, quat=q_l)

  add_bin_geoms(b, num_cans)

  # 右手首の、箱の座標での姿勢（weld の relpose）。箱は世界の向き
  rel_pos = d.xpos[rw] - center_w
  rel_q = np.zeros(4)
  mujoco.mju_mat2Quat(rel_q, d.xmat[rw].reshape(3, 3).flatten())
  eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name="bin_right_hand",
                         name1="reject_bin", name2="right_wrist_yaw_link", objtype=mujoco.mjtObj.mjOBJ_BODY)
  # data の並び：anchor(3), relpose の位置(3), relpose の四元数(4), torquescale(1)
  eq.data[0:3] = [0, 0, 0]
  eq.data[3:6] = rel_pos
  eq.data[6:10] = rel_q
  eq.data[10] = 1.0


def get_spec_with_bin(num_cans: int = 3):
  def fn() -> mujoco.MjSpec:
    spec = get_spec()
    add_reject_bin(spec, num_cans)
    return spec

  return fn


def add_bin_geoms(b, num_cans: int, contype: int = 0, conaffinity: int = 1, condim: int = 1, friction=None) -> None:
  """回収箱（胴・フランジ）と缶の geom を、箱の body b（原点＝外形の中心）に足す。"""
  ox, oy, oz = (v / 2 for v in BIN_OUTER)
  bx = BIN_INNER[0] / 2 + BIN_WALL
  by = BIN_INNER[1] / 2 + BIN_WALL
  t, ft = BIN_WALL, BIN_FLANGE_T
  col = dict(contype=contype, conaffinity=conaffinity, condim=condim)
  if friction is not None:
    col["friction"] = list(friction)
  parts = {
    # 胴（底と 4 枚の壁）
    "bin_bottom": ((bx, by, t / 2), (0, 0, -oz + t / 2)),
    "bin_wall_front": ((t / 2, by, oz), (bx - t / 2, 0, 0)),
    "bin_wall_back": ((t / 2, by, oz), (-bx + t / 2, 0, 0)),
    "bin_wall_left": ((bx, t / 2, oz), (0, by - t / 2, 0)),
    "bin_wall_right": ((bx, t / 2, oz), (0, -by + t / 2, 0)),
    # 口のフランジ（取っ手）：左右（手の側）と前後に、外形いっぱいまで張り出す
    "bin_flange_left": ((ox, (oy - by) / 2, ft / 2), (0, (oy + by) / 2, oz - ft / 2)),
    "bin_flange_right": ((ox, (oy - by) / 2, ft / 2), (0, -(oy + by) / 2, oz - ft / 2)),
    "bin_flange_front": (((ox - bx) / 2, by, ft / 2), ((ox + bx) / 2, 0, oz - ft / 2)),
    "bin_flange_back": (((ox - bx) / 2, by, ft / 2), (-(ox + bx) / 2, 0, oz - ft / 2)),
  }
  vol = {k: 8 * h[0] * h[1] * h[2] for k, (h, _) in parts.items()}
  total = sum(vol.values())
  for name, (half, pos) in parts.items():
    _add_box(b, name, half, pos, BIN_CASE_MASS * vol[name] / total, BIN_BLUE, **col)
  for i in range(num_cans):
    y = (i - (num_cans - 1) / 2) * (2 * CAN_RADIUS + 0.008)
    pos = (0.0, y, -oz + t + CAN_HALF_HEIGHT)
    b.add_geom(name=f"can{i}_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CAN_RADIUS, CAN_HALF_HEIGHT, 0],
               pos=list(pos), mass=CAN_MASS, rgba=CAN_RGBA, group=3, **col)
    b.add_geom(name=f"can{i}_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CAN_RADIUS, CAN_HALF_HEIGHT, 0],
               pos=list(pos), density=0, rgba=CAN_RGBA, group=2, contype=0, conaffinity=0)


def bin_layout(spec: mujoco.MjSpec | None = None) -> dict:
  """抱える姿勢（胴体の原点 (0, 0, z0)、向き 0）での回収箱の置き場所を計算する。

  返す値：center_w（箱の原点＝外形の中心の世界座標）、pos_l / q_l（左手首の座標での箱の位置・向き。箱は世界の向き）、
  center_root（胴体の原点から見た箱の原点）、data（MjData）、lw / rw（左右の手首の body id）
  """
  import numpy as np

  if spec is None:
    spec = get_spec()

  m = spec.compile()
  d = mujoco.MjData(m)
  d.qpos[:] = _pose_qpos(m)
  mujoco.mj_kinematics(m, d)
  lw, rw = m.body("left_wrist_yaw_link").id, m.body("right_wrist_yaw_link").id
  ox, oy, oz = (v / 2 for v in BIN_OUTER)  # 外形の半分
  hp = np.concatenate([_hand_points(m, d, "left"), _hand_points(m, d, "right")])
  hand_mid_x = (hp[:, 0].min() + hp[:, 0].max()) / 2
  y_mid = (d.xpos[lw][1] + d.xpos[rw][1]) / 2
  # フランジの下に入る部分（左右が外形の内側、前後がフランジの範囲）の手の甲のいちばん高いところ
  under = hp[(np.abs(hp[:, 1] - y_mid) <= oy) & (np.abs(hp[:, 0] - hand_mid_x) <= ox)]
  hand_top = under[:, 2].max()
  # 胴の外形：内寸＋壁。フランジは胴の上端から外形いっぱいまで張り出す
  bx = BIN_INNER[0] / 2 + BIN_WALL
  by = BIN_INNER[1] / 2 + BIN_WALL
  t, ft = BIN_WALL, BIN_FLANGE_T
  # 箱の原点 = 外形の中心。フランジの下面（上端 − ft）を手の甲の高さに合わせる
  center_w = np.array([hand_mid_x, y_mid, hand_top + ft - oz])

  R_lw = d.xmat[lw].reshape(3, 3)
  pos_l = R_lw.T @ (center_w - d.xpos[lw])
  q_l = np.zeros(4)
  mujoco.mju_mat2Quat(q_l, R_lw.T.flatten())
  root = d.xpos[m.body("base_link").id].copy()
  return {"data": d, "lw": lw, "rw": rw, "center_w": center_w, "pos_l": pos_l, "q_l": q_l,
          "center_root": center_w - root}
