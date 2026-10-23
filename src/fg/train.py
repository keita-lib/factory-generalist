"""Luna のタスクを登録してから、mjlab の学習スクリプトを呼ぶ。

使い方（リポジトリの直下で）:
  python src/fg/train.py FG-Velocity-Flat-Luna --env.scene.num-envs 4096

環境変数（追加学習のとき）:
  FG_STEP20      1 なら段の高さを 20 cm に固定し、地形のカリキュラムを止める（統合と同じ段だけで練習する。
                 run29 は追加学習の始めにカリキュラムが易しい段からやり直しになり、上りで転ぶようになった）
  FG_RESUME_LR   再開した後に、オプティマイザの学習率をこの値に置き直す（rsl_rl は保存された学習率で上書きするため）
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fg.velocity_flat  # noqa: E402,F401  タスクの登録
import fg.velocity_steps  # noqa: E402,F401  Phase 1 のタスクの登録
import fg.carry_steps  # noqa: E402,F401  落ちる箱を持つタスクの登録
import fg.pick_learn  # noqa: E402,F401  取り上げ・置くの学習（V16）の登録
import fg.turn_skill  # noqa: E402,F401  回れ右の方策の登録
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.scripts.train import main  # noqa: E402

if os.environ.get("FG_STEP20") == "1":
  _init = ManagerBasedRlEnv.__init__

  def _step20_init(self, cfg, *a, **kw):
    """環境を作る直前に、段の高さを 20 cm に固定し、地形のカリキュラムを外す。"""
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.curriculum.pop("terrain_levels", None)
      tg = cfg.scene.terrain.terrain_generator
      tg.curriculum = False
      tg.sub_terrains = {k: (replace(v, step_height_range=(0.20, 0.20)) if hasattr(v, "step_height_range") else v)
                         for k, v in tg.sub_terrains.items()}
      print(f"段 20 cm 固定・カリキュラムなし：{ {k: getattr(v, 'step_height_range', None) for k, v in tg.sub_terrains.items()} }", flush=True)
    _init(self, cfg, *a, **kw)

  ManagerBasedRlEnv.__init__ = _step20_init

if os.environ.get("FG_RESUME_LR"):
  from rsl_rl.algorithms.ppo import PPO

  _load = PPO.load
  _lr = float(os.environ["FG_RESUME_LR"])

  def _load_lr(self, *a, **kw):
    out = _load(self, *a, **kw)
    for g in self.optimizer.param_groups:
      g["lr"] = _lr
    self.learning_rate = _lr
    print(f"学習率を {_lr:g} に置き直した", flush=True)
    return out

  PPO.load = _load_lr

if __name__ == "__main__":
  main()
