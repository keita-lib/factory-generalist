"""Factory Generalist 用の段差の地形（mjlab 1.6.0）。

計画書 6.1：作業者用の 2 段のステップ、各段 200〜250 mm（初期検討値）、その上に排斥箱へ手が届く台。
ここでは mjlab のピラミッド階段（BoxPyramidStairsTerrainCfg）の外周を平らな床で埋めて、
「平らな床 → 1 段目 → 上の台（2 段目）→ 下りる」の形にしている（四方どこからでも上り下りできる）。
mjlab のピラミッド階段は「段の輪の数 + 1」だけ上がる（中央の台の上面が (輪の数 + 1) × 段の高さ）。
最初は輪を 2 にして 3 段上がる形になっていたので、輪を 1 にして、床から台まで 2 段にした。

設計の値
- 踏面の奥行き 0.35 m：Luna の足の当たり判定の長さ 180 mm（worklog 7.）の約 2 倍
- 上の台 2.0 m 四方：台の上で向きを変えられる広さ（排斥箱を取って向きを変え、同じステップを下りる想定）
- 段の高さ：難しさ（0〜1）に応じて 0.05〜0.25 m。カリキュラムで低い段から始める
"""

from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

TILE_SIZE = 8.0  # 1 区画の大きさ [m]
STEP_WIDTH = 0.35  # 踏面の奥行き [m]
PLATFORM_WIDTH = 2.0  # 上の台の一辺 [m]
NUM_RINGS = 1  # 段の輪の数。床から台の上面までの段数は NUM_RINGS + 1 = 2
STEP_HEIGHT_RANGE = (0.05, 0.25)  # 段の高さ [m]（難しさで補間）

# 輪がちょうど NUM_RINGS 個になるように、外周の平らな床の幅を決める
# 輪の数 = (区画 − 2×外周 − 台) / (2×踏面) なので、外周 = (区画 − 台 − 2×輪の数×踏面) / 2
_BORDER = (TILE_SIZE - PLATFORM_WIDTH - 2 * NUM_RINGS * STEP_WIDTH) / 2  # = 2.65 m

TWO_STEP_PLATFORM = BoxPyramidStairsTerrainCfg(
  proportion=0.8,
  size=(TILE_SIZE, TILE_SIZE),
  border_width=_BORDER,
  step_height_range=STEP_HEIGHT_RANGE,
  step_width=STEP_WIDTH,
  platform_width=PLATFORM_WIDTH,
)

FLAT = BoxFlatTerrainCfg(proportion=0.2, size=(TILE_SIZE, TILE_SIZE))


def two_step_terrain_generator_cfg(num_rows: int = 10, num_cols: int = 20) -> TerrainGeneratorCfg:
  """行ごとに難しさ（段の高さ）が上がる、カリキュラム用の地形の格子。"""
  return TerrainGeneratorCfg(
    size=(TILE_SIZE, TILE_SIZE),
    border_width=10.0,
    num_rows=num_rows,
    num_cols=num_cols,
    curriculum=True,
    sub_terrains={"two_step": TWO_STEP_PLATFORM, "flat": FLAT},
    difficulty_range=(0.0, 1.0),
  )


# run8 用：段の高さを 10〜27 cm に寄せる（目標の 20 cm の前後で多く練習させ、25 cm 以上で余裕も見る）
# run7 までは 5〜25 cm で、カリキュラムの平均が 16 cm 前後で止まり、20 cm での練習が少なかった
STEP_HEIGHT_RANGE_HARD = (0.10, 0.27)
TWO_STEP_PLATFORM_HARD = BoxPyramidStairsTerrainCfg(
  proportion=0.85,
  size=(TILE_SIZE, TILE_SIZE),
  border_width=_BORDER,
  step_height_range=STEP_HEIGHT_RANGE_HARD,
  step_width=STEP_WIDTH,
  platform_width=PLATFORM_WIDTH,
)


def two_step_hard_terrain_generator_cfg(num_rows: int = 10, num_cols: int = 20) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(TILE_SIZE, TILE_SIZE),
    border_width=10.0,
    num_rows=num_rows,
    num_cols=num_cols,
    curriculum=True,
    sub_terrains={"two_step": TWO_STEP_PLATFORM_HARD, "flat": BoxFlatTerrainCfg(proportion=0.15, size=(TILE_SIZE, TILE_SIZE))},
    difficulty_range=(0.0, 1.0),
  )
