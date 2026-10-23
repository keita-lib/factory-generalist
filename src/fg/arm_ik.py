"""腕の逆運動学（取り上げ・置く動作、箱を挟む姿勢）。

腕の付け根は上体（waist_pitch_link）なので、上体の座標で解けば、脚・腰の姿勢によらず使える。
目標は「箱の座標の、抱える姿勢からのずれ」で与える：
  箱の座標を上体の座標で (dx, dy, dz) 動かし、上体の z 軸まわりに dyaw 回す。さらに左右の手を spread だけ外へ（負なら内へ）
→ 左右の手首（wrist_yaw_link）の目標の位置・向き → 肩 3 軸・肘・手首 yaw の 5 関節を減衰最小二乗で解く（位置を重く、向きを軽く）
"""

import re

import mujoco
import numpy as np

from fg.luna_constants import CARRY_ARM_POSE, KNEES_BENT_KEYFRAME, bin_layout

TORSO = "waist_pitch_link"
SIDES = ("left", "right")


def _joint_names(side: str) -> list[str]:
  return [f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint", f"{side}_shoulder_yaw_joint",
          f"{side}_elbow_joint", f"{side}_wrist_yaw_joint"]


class ArmIK:
  def __init__(self):
    L = bin_layout()
    from fg.free_bin import get_spec_with_hooks

    self.m = get_spec_with_hooks().compile()
    self.d = mujoco.MjData(self.m)
    m, d = self.m, self.d
    d.qpos[2] = KNEES_BENT_KEYFRAME.pos[2]
    d.qpos[3:7] = [1, 0, 0, 0]
    for j in range(1, m.njnt):
      for pat, v in {**KNEES_BENT_KEYFRAME.joint_pos, **CARRY_ARM_POSE}.items():
        if re.fullmatch(pat, m.joint(j).name):
          d.qpos[m.jnt_qposadr[j]] = v
    self.q0 = d.qpos.copy()
    mujoco.mj_kinematics(m, d)
    self.torso = m.body(TORSO).id
    Rt, pt = d.xmat[self.torso].reshape(3, 3).copy(), d.xpos[self.torso].copy()
    self.Rt, self.pt = Rt, pt
    # 抱える姿勢での箱の原点（上体の座標）と、手首の位置・向き（上体の座標）
    self.bin_t = Rt.T @ (L["center_w"] - pt)
    self.wrist = {}
    for side in SIDES:
      b = m.body(f"{side}_wrist_yaw_link").id
      self.wrist[side] = (Rt.T @ (d.xpos[b] - pt), Rt.T @ d.xmat[b].reshape(3, 3))

  def targets(self, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0, spread=0.0, droll=0.0, side_off=None, drot=None):
    """手首の目標（上体の座標）：{side: (位置, 回転行列)}。droll：箱の左右の傾き（上体の x 軸まわり）
    drot：箱の回転のずれ（上体の座標の回転ベクトル [rad]）。動かした後の箱の中心のまわりに、両手の目標をまとめて回す"""
    c, s = np.cos(dyaw), np.sin(dyaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # 上体が左右に傾いていると、箱は上体から見て傾いて見える。左右の手の高さを、その傾きに合わせる
    cr, sr = np.cos(droll), np.sin(droll)
    Rz = Rz @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    out = {}
    for side, sgn in zip(SIDES, (1.0, -1.0)):
      p, R = self.wrist[side]
      p = p + np.array([0.0, sgn * spread, 0.0])
      p = self.bin_t + Rz @ (p - self.bin_t) + np.array([dx, dy, dz])
      if side_off is not None:
        p = p + np.asarray(side_off[side])  # 左右それぞれの補正（上体の座標）
      Rs = Rz @ R
      if drot is not None:
        c0 = self.bin_t + np.array([dx, dy, dz])
        v = np.asarray(drot, dtype=float)
        ang = float(np.linalg.norm(v))
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, v / ang if ang > 0 else np.array([1.0, 0.0, 0.0]), ang)
        Rv = np.zeros(9)
        mujoco.mju_quat2Mat(Rv, q)
        Rv = Rv.reshape(3, 3)
        p = c0 + Rv @ (p - c0)
        Rs = Rv @ Rs
      out[side] = (p, Rs)
    return out

  def solve(self, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0, spread=0.0, iters=600, w_ori=0.5, w_pitch=0.0, w_x=1.0, droll=0.0, side_off=None, drot=None):
    """関節名 → 角度 [rad]、最大の位置誤差 [m]、最大の向きの誤差 [度] を返す。"""
    m, d = self.m, self.d
    d.qpos[:] = self.q0
    tg = self.targets(dx, dy, dz, dyaw, spread, droll, side_off, drot)
    result, worst, worst_o = {}, 0.0, 0.0
    for side in SIDES:
      names = _joint_names(side)
      jids = [m.joint(n).id for n in names]
      dofs = [m.jnt_dofadr[j] for j in jids]
      qadr = [m.jnt_qposadr[j] for j in jids]
      b = m.body(f"{side}_wrist_yaw_link").id
      p_t, R_t = tg[side]
      p_w, R_w = self.pt + self.Rt @ p_t, self.Rt @ R_t
      for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        ep = p_w - d.xpos[b]
        Rc = d.xmat[b].reshape(3, 3)
        Re = R_w @ Rc.T
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Re.flatten())
        eo = np.zeros(3)
        mujoco.mju_quat2Vel(eo, q, 1.0)
        # 手の前後の傾き（上体の y 軸まわり）は合わせない：肩・肘の 2 関節では、上下に動かしつつ手の傾きは保てないため
        # w_pitch > 0 なら前後の傾きも合わせる（そのぶん前後の位置 x を w_x で軽くして、位置と傾きを両立させる）
        ay = self.Rt[:, 1]
        ax = self.Rt[:, 0]
        eo = eo - ay * (eo @ ay) * (1.0 - w_pitch)
        ep = ep - ax * (ep @ ax) * (1.0 - w_x)
        if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(eo) < 2e-3:
          break
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, b)
        jr = jr - np.outer(ay, ay @ jr) * (1.0 - w_pitch)
        jp = jp - np.outer(ax, ax @ jp) * (1.0 - w_x)
        J = np.vstack([jp[:, dofs], w_ori * jr[:, dofs]])
        e = np.concatenate([ep, w_ori * eo])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        for k, a in enumerate(qadr):
          lo, hi = m.jnt_range[jids[k]]
          d.qpos[a] = np.clip(d.qpos[a] + 0.7 * dq[k], lo * 0.98, hi * 0.98)
      mujoco.mj_kinematics(m, d)
      worst = max(worst, float(np.linalg.norm(p_w - d.xpos[b])))
      worst_o = max(worst_o, float(np.degrees(np.linalg.norm(eo))))
      # 前後の傾き（上体の y 軸まわり）がどれだけ変わったか
      Rc = d.xmat[b].reshape(3, 3)
      self.last_pitch = float(np.degrees(np.arctan2(*(R_w.T @ Rc)[[0, 0], [2, 0]])))
      for n, a in zip(names, qadr):
        result[n] = float(d.qpos[a])
    return result, worst, worst_o

  def rel_jacobian(self, names, h_pos=0.01, h_rot=0.03, **kw):
    """箱の上体に対するずれ δ（位置 3 [m]・回転ベクトル 3 [rad]、上体の座標）に対する関節角の 1 次の変化：(関節数 × 6)。
    置くとき、実際の上体と箱の姿勢から腕の目標を直すのに使う（2026-10-17）。中心差分"""
    J = np.zeros((len(names), 6))
    for k in range(6):
      col = []
      for sg in (1.0, -1.0):
        kk = dict(kw)
        if k < 3:
          key = ("dx", "dy", "dz")[k]
          kk[key] = kk.get(key, 0.0) + sg * h_pos
        else:
          v = np.zeros(3)
          v[k - 3] = sg * h_rot
          kk["drot"] = v
        r, _, _ = self.solve(**kk)
        col.append(np.array([r[nm] for nm in names]))
      J[:, k] = (col[0] - col[1]) / (2 * (h_pos if k < 3 else h_rot))
    return J


if __name__ == "__main__":
  ik = ArmIK()
  for kw in ({}, {"spread": -0.01}, {"dz": -0.055}, {"dz": -0.055, "spread": 0.03}, {"dz": 0.02}, {"dx": 0.03, "dyaw": 0.1}):
    r, w, wo = ik.solve(**kw)
    print(kw, f"誤差 {w * 1000:.1f} mm・{wo:.1f} 度、前後の傾き {ik.last_pitch:+.1f} 度", {k: round(v, 3) for k, v in r.items() if k.startswith("left")})
