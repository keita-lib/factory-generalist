# Factory Generalist：ヒューマノイドによる排斥回収ケースの運搬（シミュレーション）

ヒューマノイド LimX Luna（27 自由度、マニピュレータ機能非搭載の腕）を、MuJoCo ベースの学習環境 [mjlab](https://github.com/mujocolab/mjlab) でシミュレーションし、缶飲料ラインの検査室を模した区画で次の作業をさせる。

検査テーブルから 5 m 歩く → 20 cm × 2 段のステップを昇る → 排斥テーブルから回収ケースを取り上げる → 振り返ってステップを降りる → 検査テーブルに置く。これを 3 往復。人間のデモンストレーションデータは使っていない。

- 歩く・段差・運搬：強化学習（PPO）で学習した方策（run14）。ステップを昇る間だけ、段 20 cm で追加学習した方策（run30）に切り替える
- 取り上げ：強化学習で学習した方策（run28）
- 置いた後の振り返り（その場で 180 度回る）：「回れ右」だけを追加学習した方策（run31）
- 置く・作業の順番：シーケンス制御（`src/fg/integrate.py`）。置くは、その手順の上に強化学習の小さな補正と開くタイミングを足す（残差強化学習、`src/fg/place_residual.py`）
- 結果（1024 台の並列評価 × 乱数シード 3 つ）：**3 往復の完了 97.3 %（2989/3072）、3 往復 167 秒**（残差強化学習なしは 96.2 %・179 秒）

地形と箱の位置はシミュレーターの正しい値を使っている（カメラなどの知覚は使っていない）。実機では試していない。

## 動かした PC

| 項目 | 内容 |
|---|---|
| OS | Windows 11 Pro |
| CPU | AMD Ryzen 9 7900X3D（12 コア 24 スレッド） |
| GPU | NVIDIA GeForce RTX 3090（24 GB）、ドライバ 610.88 |
| メモリ | 64 GB |
| Python | 3.12.6（venv は uv で作成） |
| 主なパッケージ | mjlab 1.6.0、mujoco / mujoco-warp 3.11.0、warp-lang 1.18.0、torch 2.11.0+cu128、rsl-rl-lib 5.4.2 |

- 学習・評価とも GPU（CUDA）を使う。学習は 4096 台、評価は 1024 台を同時に動かした。
- 学習の時間の目安（RTX 3090）：取り上げの追加学習 run28（800 回）が約 1 時間、歩行の run14（2000 回）が約 1 時間。
- GPU のメモリ（VRAM）の使用量（何も動かしていない時との差、Windows で測定）：学習（段差・運搬のタスク、4096 台）で約 11.5 GB、統合の評価（1024 台）で約 4.2 GB。学習の 1 回は約 1.9 秒。
- GPU のメモリが少ない場合は、台数（`--num-envs`、学習は `--env.scene.num-envs`）を減らす。
- Linux（Ubuntu）でも mjlab は動くが、この PC 以外では試していない。

## 準備

1. このリポジトリを clone する。
2. Luna のロボットモデル（LimX Dynamics 公式、Apache-2.0）を `third_party/` に置く。動作を確かめたのはコミット `30431c9`。

   ```bash
   git clone https://github.com/LimXDynamics/luna-description.git third_party/luna-description
   git -C third_party/luna-description checkout 30431c9
   ```

3. Python 3.12 の venv を作り、パッケージを入れる（[uv](https://docs.astral.sh/uv/) を使う例。pip でも同じ）。

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   uv pip install --python .venv -r requirements.txt
   ```

## 動かす（学習済みの方策で 3 往復）

学習済みの方策は `checkpoints/` に入っている（学習し直さなくても動かせる）。

| ファイル | 中身 |
|---|---|
| `checkpoints/loco_run14_model_24493.pt` | 歩く・段差・運搬の方策（run14） |
| `checkpoints/pick_run28_model_30500.pt` | 取り上げの方策（run28） |
| `checkpoints/climb_run30_model_24692.pt` | ステップを昇る間だけ使う歩行の方策（run30：run14 を段 20 cm 固定で 200 回追加学習） |
| `checkpoints/turn_v1_model_24700.pt` | 回れ右の方策（run31。run14 を「その場で回る」指令だけで 300 回追加学習） |
| `checkpoints/place_residual_res2.pt` | 置くの残差の方策（決まった手順の腕の目標への補正 ±0.04 rad と、開くタイミング。PPO で 67 回、続けて失敗の罰 −30 で 50 回） |

```bash
# 3 往復の完了率を測る（1024 台、乱数シード 1）。Windows は Git Bash で
# Linux は .venv/bin/python を使う（スクリプトが自動で選ぶ）
bash scripts/integrate_best.sh 1024 3 1

# まず小さく試す（16 台、1 往復）
bash scripts/integrate_best.sh 16 1 1

# 完了した 1 台を動画にする（32 台、3 往復、乱数シード 21。out/ は先に作っておく）
bash scripts/integrate_best.sh 32 3 21 --replay-video out/3trips.mp4
```

終わると、完了した台の数と、失敗の理由（転倒・箱の落下・時間切れの内訳）を表示する。`--metrics <ファイル.json>` を足すと、段階ごとの時間・ほぼ止まっている時間・関節の加速度と躍度・仕事・箱の加速度も表示する。引数の意味は `python src/fg/integrate.py --help`。

## 学習し直す

学習は `src/fg/train.py <タスク名>` で行う（mjlab の学習スクリプトに、Luna のタスクを登録してから渡す）。

```bash
.venv/Scripts/python.exe src/fg/train.py FG-Velocity-Flat-Luna --env.scene.num-envs 4096
```

最終の方策までは、前の方策の重みから続けて学習を重ねている。

| 段階 | タスク名 | run | 内容 |
|---|---|---|---|
| 平らな床を歩く | `FG-Velocity-Flat-Luna`（-Curriculum・-Fast・-Gait） | run1〜run4 | 速度の指令に従って歩く（入力 93） |
| ステップを昇り降りする | `FG-Velocity-Steps-Luna`（-NoStall・-V7〜V10） | run5b〜run10 | 足元の地形の高さ 187 点を入力に足す（入力 280）。run5b はゼロから学習 |
| 箱を持って運ぶ | `FG-Carry-Steps-Luna-V11`〜`V15` | run11〜run15 | 受け具との接触だけで支える箱を持って歩く。run14 で、学習中に腕を動かし続ける |
| 取り上げる | `FG-Pick-Place-Luna-V16` | run16〜run28 | 腕も方策で動かす（入力 302）。run14 の重みから引き継ぐ |
| 回れ右 | `FG-Turn-Luna-V1` | run31 | 前後・横の指令 0、回る ±1.2 rad/s だけで、run14 から 500 回（採用は 300 回目）。`src/fg/turn_skill.py` |
| ステップを昇る | `FG-Carry-Steps-Luna-V14` | run30 | 段 20 cm 固定・カリキュラムなしで、run14 から 200 回 |

回れ右と run30 の学習（どちらも run14 から、学習率を置き直して固定）：

```bash
# 回れ右（学習率 1e-4、500 回）
FG_RESUME_LR=1e-4 .venv/Scripts/python.exe src/fg/train.py FG-Turn-Luna-V1 --env.scene.num-envs 4096   --agent.max-iterations 500 --agent.resume True --agent.load-run <run14 のフォルダ> --agent.load-checkpoint model_24493.pt   --agent.algorithm.schedule fixed --agent.algorithm.learning-rate 1e-4
# run30（段 20 cm 固定、学習率 5e-5、200 回）
FG_STEP20=1 FG_RESUME_LR=5e-5 .venv/Scripts/python.exe src/fg/train.py FG-Carry-Steps-Luna-V14 --env.scene.num-envs 4096   --agent.max-iterations 200 --agent.resume True --agent.load-run <run14 のフォルダ> --agent.load-checkpoint model_24493.pt   --agent.algorithm.schedule fixed --agent.algorithm.learning-rate 5e-5 --agent.algorithm.entropy-coef 0.001
```

run30 は全体に使うと、寄る・下がるときの小さな指令に従わなくなった（1 往復 71.7〜82.5 %）。昇る間だけに使う。

置くの残差強化学習（`src/fg/place_residual.py`）は、統合のスクリプトの中で学習する。まず最良の設定で「置くに入った瞬間」の状態を集め、
全台をその状態から始めて「置く → 0.5 m 下がる」を繰り返す（1024 台、約 13 秒で 1 回）：

```bash
# 置くに入った瞬間の状態を集める（3 往復、約 3000 件）
bash scripts/integrate_1017.sh 1024 3 11 --entry-log out/bank_s11.pt
# 学習（0.1 秒ごとに判断。報酬：成功 +10、失敗 −10、時間 −1/秒、補正の大きさ・箱の傾きに小さな罰）
bash scripts/integrate_1017.sh 1024 1 1 --start-place out/bank_s11.pt --residual-train out/res1 --residual-iters 200
```

成功率は 30 回目ごろから横ばいになったので、67 回目（res1）で止めた。続けて失敗の罰を重くし（`--residual-policy <res1> --residual-fail-w 30`）、その 50 回目を使っている（res2。3 往復 96.8 → 97.3 %）。

学習を続けるときは、mjlab の引数 `--agent.resume True --agent.load-run <フォルダ> --agent.load-checkpoint <ファイル>` を使う。ログとチェックポイントは `logs/rsl_rl/` に書き出される（大きいのでリポジトリには入れていない）。取り上げの学習（run28）では、統合の評価で記録した「取り上げを始めた瞬間の状態」（`integrate.py --entry-log`）を、環境変数 `FG_ENTRY_BANK` で練習の始めの場面に混ぜている。

## 構成

| 場所 | 中身 |
|---|---|
| `src/fg/` | タスクの定義（`velocity_*`・`carry_steps`・`pick_learn`・`turn_skill`）、置くの残差強化学習（`place_residual`）、ロボットの設定（`luna_constants`）、回収ケースと受け具（`free_bin`）、段階の進行と評価（`integrate`） |
| `scripts/` | 最良の設定で統合を走らせるスクリプト（`integrate_best.sh`）と、残差強化学習の手順で使う設定（`integrate_1017.sh`） |
| `checkpoints/` | 学習済みの方策 |
| `third_party/` | Luna のロボットモデル（各自で clone、リポジトリには入れない） |

## ライセンス・出典

- ロボットモデル：[LimXDynamics/luna-description](https://github.com/LimXDynamics/luna-description)（Apache-2.0）
- 学習環境：[mjlab](https://github.com/mujocolab/mjlab)、[MuJoCo](https://github.com/google-deepmind/mujoco)、[rsl_rl](https://github.com/leggedrobotics/rsl_rl)
