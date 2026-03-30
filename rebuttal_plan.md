# Rebuttal Plan 

## Summary
- Q3 主线固定为 **best-source selection / source ranking**，不是正负迁移判别。
- Q4 主线升级为一个 **Exp 9 风格但 proxy-compatible 的 data curation toolkit**：
  - 不再使用 literal `pruning + graphon-matched merging + curriculum`
  - 改为 **pruning + large-graph reweighting + standard merging**
  - 三种干预都能在训练前通过 `D_total` 打分并闭环选择
- 主文只使用 **total train-target token discrepancy** 作为经验 proxy，不声称精确估计 `ε₁ + ε_gra + ε₂`。
- 主实验固定为三个：
  - **Exp A: Synthetic best-source selection**
  - **Exp B: Mini real best-source selection**
  - **Exp C: Metric-guided intervention selection under target shift**
- 若 Exp C 不稳定，fallback 才是 merge-only ratio selection；不是默认主线。

## Key Protocol Decisions
- 主模型固定为 `DeepSets + Eig-PE (k=32)`。
- `GIN + Eig-PE (k=32)` 只做 synthetic robustness，不阻塞主 rebuttal。
- 所有主实验统一 `3 seeds = {0,1,2}`，报告 `mean ± std`。
- `D_total` 一律使用现有 proportional token discrepancy。
- 所有 selection 都禁止使用最终 test labels。

## Q3: Transfer Harness
- Source 端：
  - 在 source dataset 上监督训练 `encoder + source head`
  - real 数据 source 预算固定为 `64 graphs/class train + 16 graphs/class val`
  - source checkpoint 只由 source val 选择
- Target 端：
  - 丢弃 source head，冻结 encoder，只训练新的 **linear probe**
  - `probe_train = 8 graphs/class`
  - `probe_val = 8 graphs/class`
  - `selection_pool = target val split`，仅用于 discrepancy，不参与 probe 训练
  - `final_test = target test split`
- Scratch baseline：
  - 相同 few-shot target 预算下，从随机初始化训练 target encoder+head
- Primary Q3 metrics：
  - `top-1 source hit rate`
  - `top-1 regret = selected source error - oracle best-source error`
  - `Spearman ρ(D_total, transferred test error)`
- Tie-break 规则：
  - 若多个 source 的 `D_total` 相同，选 source train size median 更接近 target 的那个
  - 若仍相同，按数据集名字典序固定

## Q4: Exp 9-Calibrated Toolkit
- Q4 不再用 literal curriculum，因为它不改变 train token 分布，无法在训练前被 discrepancy proxy 打分。
- Q4 的候选干预固定为 5 个 candidate：
  - `vanilla`
  - `prune10`
  - `reweight-large`
  - `merge1`
  - `merge3`
- 每个 candidate 的定义固定：
  - `vanilla`：原始 train set
  - `prune10`：按类计算每个图的 graph signature `mean(tokens)`；与类 centroid 的 `L2` 距离排序，删除每类最远的 `10%`
  - `reweight-large`：每类中 size 大于等于该类 `75th percentile` 的训练图复制一次，形成 2x effective weight
  - `merge1`：`usvt`, `merging_size = 2.0`, `merging_ratio = 1%`
  - `merge3`：`usvt`, `merging_size = 2.0`, `merging_ratio = 3%`
- Q4 数据协议固定为新的 **four-way size-gap split**：
  - `small_train`
  - `small_val`
  - `large_selection`
  - `large_test`
- split 规则固定：
  - 先按 size 排序，找到满足 non-overlap 且 median ratio 最接近 `2.0` 的切点
  - 小图侧再按 label stratify 成 `90% train / 10% val`
  - 大图侧再按 label stratify 成 `50% selection / 50% test`
- Q4 selection 规则：
  - 对 5 个 candidate 分别构造训练集
  - 计算 `D_total(candidate_train, large_selection)`
  - 选 `argmin D_total` 作为 prescribed candidate
- Q4 oracle 与对照：
  - `oracle = 5 个 candidate 中 final large_test error 最低者`
  - 主要比较：
    - `metric-guided`
    - `vanilla`
    - `oracle`
- Primary Q4 metrics：
  - `oracle gap = selected test error - oracle test error`
  - `harmful picks = # {selected test error > vanilla test error}`
  - `mean improvement over vanilla`
- Q4 数据集固定为 `COLLAB`, `IMDB-BINARY`, `REDDIT-BINARY`

## Experiments
- **Exp A: Synthetic best-source selection**
  - 使用 controlled Fourier graphon
  - 5 个 domain，perturbation levels 固定为 `{0.0, 0.2, 0.4, 0.6, 0.8}`
  - 共 `20` 个 ordered pairs
  - 输出：selected source、oracle source、top-1 regret、`ρ`
- **Exp B: Mini real best-source selection**
  - 数据集固定为 `COLLAB`, `IMDB-BINARY`, `REDDIT-BINARY`, `PROTEINS`
  - 共 `12` 个 ordered pairs
  - 主文措辞固定为 `small-scale real representation transfer benchmark`
  - 输出：per-target selected source、oracle source、top-1 hit rate、mean regret
- **Exp C: Metric-guided intervention selection**
  - 3 个真实数据集上运行 5-candidate toolkit
  - 输出：每个 dataset 的 prescribed candidate、oracle candidate、oracle gap、是否 harmful
  - 主 claim 固定为：
    - token discrepancy can prescribe a data curation action
    - and selected actions are competitive with oracle under target shift
- 可选 appendix：
  - synthetic intervention sanity check，验证 outlier-heavy / large-size-gap / graphon-shift 条件下 candidate 偏好是否符合直觉
  - bootstrap `eps1_proxy / eps2_proxy / eps_gra_proxy`
  - GIN synthetic robustness

## Implementation Changes
- 给 `DeepSets` 和 `GIN` 增加 `encode(...) -> graph_embedding`
- 新增 `transfer_probe` harness 和 JSON 输出
- 新增 target few-shot sampler，强制 `probe_train / probe_val / selection / test` 无重叠
- 新增 four-way size-gap split helper，专供 Q4 使用
- 新增 candidate builder：
  - pruning builder
  - large-graph duplication builder
  - merge candidate builder
- 现有 train/eval 路径保持兼容；旧实验不回归

## Test Plan
- `encode()` smoke test：shape 正确，旧 `forward` 路径不回归
- few-shot split test：`8/8/selection/test` 无重叠且 per-class 预算正确
- four-way size-gap split test：`max(small side) < min(large side)`，selection/test 分离正确
- candidate builder smoke test：5 个 Q4 candidate 都能构造并产出稳定 token sets
- `transfer_probe` smoke test：1 个 synthetic pair 跑通 source-train / probe-train / scratch baseline
- Q4 smoke test：1 个 dataset 的 5-candidate selection 跑通并产出 oracle gap
- fixed-seed reproducibility：`D_total` 与 sampled splits 在固定 seed 下可复现

## Assumptions and Defaults
- 这份内容是新的 `rebuttal_plan.md` 完整替换版。
- rebuttal 主文不再声称“精确分解 theorem 三项”；只声称 `D_total` 是一个实用 proxy。
- `prune10` 和 `reweight-large` 是为了构造 **proxy-compatible** 的 Exp 9 版本，不主张它们分别精确对应 `ε₁` 或 `ε₂`。
- 若 `PROTEINS` 无法满足固定 source budget，Exp B 自动降级为 `3-dataset case study`，但预算规则不改变。
- 若 Exp C 在 3 个数据集上全部不优于 vanilla，则主文 Q4 回退为 merge-only ratio selection，Exp 9-calibrated toolkit 移到 appendix。
