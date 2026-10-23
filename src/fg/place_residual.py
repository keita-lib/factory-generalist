"""置く動作の残差強化学習（2026-10-18）。

置くは、逆運動学の決まった手順（integrate.py の --place-closed）で腕の目標を作る。この方策は、その手順の上に
  (1) 腕の 10 関節の目標への小さな補正（足し算、最大 ±scale rad）
  (2) 「もう開いてよい」の合図（合わせる段階で、載った判定を待たずに開く）
を足す。手順はそのまま土台に残るので、補正が 0 なら今までと同じ動き。

学習は integrate.py の --residual-train で行う：全台を「置くに入った瞬間」の記録（統合の 3 往復で集めた状態）から始め、
置く → 下がる（0.5 m）を 1 回の試行にする。下がり終えて箱が天板の上にあれば成功、落とす・転ぶ・15 秒で終わらないは失敗。
判断は 0.1 秒ごと（5 歩ごと）で、その間は同じ行動を保つ。
"""
import math

import torch
import torch.nn as nn

N_ARM = 10
N_ACT = N_ARM + 1  # 腕の補正 10 ＋ 開く合図 1
OPEN_BIAS = -2.5  # 開く合図の始めの平均（標準偏差 1.0 のとき、0.2 秒ごとの判断で 0.6 % が開く側）


def build_obs(ph, in_back, tau, t_ph, hook_err, gap, bspd, tilt, dxy, rel6, arm_err, arm_vel, lin_b, ang_b, grav, body_tab, box_tab, sel):
  """方策の入力（台数 × 59）。単位をそろえるため、長さは cm 程度、角度は rad のまま"""
  oh = torch.nn.functional.one_hot(ph.clamp(0, 4), 5).float()
  return torch.cat([
    oh, in_back[:, None].float(),
    (tau / 10.0)[:, None], (t_ph.clamp(0, 5) / 5.0)[:, None],
    hook_err.reshape(-1, 6) * 10.0,
    gap[:, None] * 10.0, bspd[:, None], tilt[:, None], dxy[:, None] * 10.0,
    rel6[:, :3] * 10.0, rel6[:, 3:] * 5.0,
    arm_err, arm_vel * 0.1,
    lin_b, ang_b, grav,
    body_tab * 5.0, box_tab * 5.0,
    sel.float() * 0.5,
  ], dim=-1)


N_OBS = 5 + 1 + 2 + 6 + 4 + 6 + N_ARM + N_ARM + 9 + 2 + 2 + 2


class RunningNorm(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.register_buffer("mean", torch.zeros(n))
    self.register_buffer("var", torch.ones(n))
    self.register_buffer("count", torch.tensor(1e-4))

  def update(self, x):
    bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    d = bm - self.mean
    tot = self.count + bc
    self.mean += d * bc / tot
    self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
    self.count = tot

  def forward(self, x):
    return ((x - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-5, 5)


def mlp(i, o, h=128):
  return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class ResidualAC(nn.Module):
  def __init__(self, arm_std=0.3, open_std=1.0):
    super().__init__()
    self.norm = RunningNorm(N_OBS)
    self.actor = mlp(N_OBS, N_ACT)
    self.critic = mlp(N_OBS, 1)
    # 始めは補正 0（決まった手順そのまま）、開く合図はほぼ出さない
    nn.init.zeros_(self.actor[-1].weight)
    with torch.no_grad():
      self.actor[-1].bias.zero_()
      self.actor[-1].bias[N_ARM] = OPEN_BIAS
    self.log_std = nn.Parameter(torch.tensor([math.log(arm_std)] * N_ARM + [math.log(open_std)]))

  def dist(self, obs):
    mu = self.actor(self.norm(obs))
    return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

  def value(self, obs):
    return self.critic(self.norm(obs)).squeeze(-1)

  @torch.no_grad()
  def act_mean(self, obs):
    return self.actor(self.norm(obs))


class PPO:
  def __init__(self, ac, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5, minibatches=4, vf_coef=1.0, max_grad=1.0):
    self.ac = ac
    self.opt = torch.optim.Adam(ac.parameters(), lr=lr)
    self.gamma, self.lam, self.clip, self.epochs, self.mb, self.vf, self.max_grad = gamma, lam, clip, epochs, minibatches, vf_coef, max_grad
    self.buf = []

  def add(self, obs, act, logp, val, rew, done):
    self.buf.append((obs, act, logp, val, rew, done))

  def update(self, last_val):
    obs, act, logp, val, rew, done = (torch.stack(x) for x in zip(*self.buf))
    T = len(self.buf)
    adv = torch.zeros_like(rew)
    gae = torch.zeros_like(last_val)
    for t in reversed(range(T)):
      nv = last_val if t == T - 1 else val[t + 1]
      nonterm = 1.0 - done[t]
      delta = rew[t] + self.gamma * nv * nonterm - val[t]
      gae = delta + self.gamma * self.lam * nonterm * gae
      adv[t] = gae
    ret = adv + val
    B = T * obs.shape[1]
    obs, act, logp, adv, ret = obs.reshape(B, -1), act.reshape(B, -1), logp.reshape(B), adv.reshape(B), ret.reshape(B)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    st = {"pi": 0.0, "vf": 0.0, "kl": 0.0}
    for _ in range(self.epochs):
      perm = torch.randperm(B, device=obs.device)
      for idx in perm.chunk(self.mb):
        d = self.ac.dist(obs[idx])
        lp = d.log_prob(act[idx]).sum(-1)
        ratio = (lp - logp[idx]).exp()
        pi = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
        vf = (self.ac.value(obs[idx]) - ret[idx]).square().mean()
        loss = pi + self.vf * vf
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad)
        self.opt.step()
        st["pi"] += float(pi) / (self.epochs * self.mb)
        st["vf"] += float(vf) / (self.epochs * self.mb)
        st["kl"] += float((logp[idx] - lp).mean()) / (self.epochs * self.mb)
    # 入力の正規化は、更新の後に進める（更新の間に変えると、集めた時の確率と比べられない）
    self.ac.norm.update(obs)
    self.buf = []
    return st
