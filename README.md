# RTP-Gate / RTD

> 一句话介绍：这是一个研究“剪掉大模型某些层以后，会不会明显伤害推理能力”的实验仓库。我们的核心想法是，先用一种便宜的诊断分数提前判断剪层风险，再决定哪些候选方案值得去跑昂贵的下游评测。

如果你是第一次接触这个项目，可以先把它理解成这样：

- 大模型有很多层。
- 我们想试着删掉其中一些层，让模型更轻、更快，或者帮助我们理解哪些层更重要。
- 但问题是：真正验证“删掉这些层会不会把模型推理能力搞坏”，通常要跑很多正式任务，成本很高。
- 所以我们设计了 **RTD** 这个分数，想先做一个“风险预警”。
- **RTP-Gate** 则是利用 RTD 分数，从很多候选剪层方案中挑出看起来更安全、或者至少更值得继续评测的方案。

这个仓库不是完整的 Paper2 复现仓库，而是从更大的工作目录里导出的一个**自研实验快照仓库**。重点不是复述原论文，而是保存我们自己的 RTP-Gate / RTD 思路、脚本、结果快照和实验组织方式。

---

## 1. 这个项目到底在研究什么？

我们关注的问题是：

**如果要对大语言模型做 layer pruning（按层剪枝），能不能在不先跑一大堆正式任务的前提下，快速判断某个剪层方案是不是危险？**

更具体一点：

- 模型有很多 Transformer 层。
- 有些层删掉以后，模型可能几乎不受影响。
- 但有些层一删，数学推理、长链条 reasoning、答案稳定性就会明显变差。
- 如果每试一个候选方案都去跑完整的 GSM8K、XSUM、分类任务，成本会很高。

所以我们提出一个实验路线：

1. 先让**完整模型**在标准题目上留下细粒度推理痕迹。
2. 再让“删过层的候选模型”在同一批样本上重放这些痕迹。
3. 观察它和完整模型相比，在哪些 token 上开始偏离。
4. 把这种偏离压缩成一个风险分数，也就是 **RTD**。
5. 用 RTD 去筛选和排序候选剪层方案，这一步就叫 **RTP-Gate**。

简单说：

- **RTD** 负责“看风险”
- **RTP-Gate** 负责“拿这个风险去挑候选”

---

## 2. 为什么要做这个项目？

做 layer pruning 时，最麻烦的一点是：**真正重要的不是模型有没有变小，而是变小以后还能不能正常推理。**

如果只看参数量、层号、或者非常粗糙的代理指标，往往不够。因为：

- 有的层虽然位置靠后，但删掉以后损伤很大。
- 有的层看起来不起眼，但对数学题或者答案校准很关键。
- 不同剪层组合之间还会有相互作用，单看单层不一定够。

所以我们想做的不是“直接宣布最佳剪枝算法”，而是先解决一个更基础、更现实的问题：

**能不能做一个便宜、可批量计算的风险诊断器，先把高风险候选排掉？**

这就是这个项目存在的意义。

---

## 3. 核心概念：用通俗话讲 RTP-Gate / RTD / dense trace / raw eval

### 3.1 什么是 layer pruning？

你可以把大模型想成一条很长的流水线，每一层都在继续加工信息。

- **layer pruning**：就是把其中一些层直接跳过或者删掉
- 目标通常是：减少计算、缩小模型、研究层的重要性

但删层以后，模型不一定“直接死掉”，它可能只是：

- 数学题更容易错
- 推理链条变短或变乱
- 变得更自信但更错
- 对摘要或分类任务的表现也出现变化

### 3.2 什么是 dense trace？

这是本项目里很关键的中间产物。

可以把它理解成：

**先让完整模型在一批标准样本上，留下它逐 token 推理时的细粒度痕迹。**

这些痕迹不是普通的“最终答案”，而是更细的过程信息，例如：

- 每一步 token 的概率分布
- 某些位置的 top-k 候选
- 哪些 token 属于数学推理段、答案段等

为什么要这样做？

因为我们后面不是只想知道“最终答对没”，而是想知道：

**删层后的模型，是从哪里开始偏离完整模型的？偏离得有多早、多明显？**

### 3.3 什么是 RTD？

RTD 可以粗略理解成：

**某个剪层候选，在重放 dense trace 时，相比完整模型偏离得有多严重。**

分数越高，通常表示风险越大。

它不是直接的任务准确率，但它想回答一个非常实用的问题：

> “如果我真的把这些层剪掉，这个候选方案是不是很可能伤到推理能力？”

所以 RTD 更像一个**风险预警分数**。

### 3.4 什么是 RTP-Gate？

RTP-Gate 不是另一个任务分数，而是一种“门控 / 筛选”思路。

它做的事情是：

- 先构造很多剪层候选
- 对这些候选算 RTD
- 用 greedy 的方式，一步步挑出风险更低、或至少更值得继续评测的方案

换句话说：

- **RTD** 是测量工具
- **RTP-Gate** 是用这个测量工具做筛选决策

### 3.5 什么是 raw eval？

raw eval 就是正式下游评测，也就是“真刀真枪地看任务表现”。

在这个项目里，主要包括：

- **GSM8K**：数学推理主证据
- **XSUM**：摘要任务，更多是辅助观察
- **7 个 classification 任务**：负对照，帮助看模型是不是整体崩坏

这一步昂贵，所以我们才想先用 RTD 做诊断。

### 3.6 什么是 consistency check？

有两种常见方式来模拟剪层：

1. **runtime skip**：运行时跳过某些层
2. **saved pruned model**：真的把模型保存成删层后的结构

两者如果表现差很多，就需要小心解释结果。

所以项目里还有 **saved-model consistency** 检查，用来确认：

**“运行时跳层看到的风险”，和“真正保存后的删层模型”是否方向一致。**

---

## 4. 这个仓库里包含什么？不包含什么？

### 4.1 仓库里包含的内容

- `experiments/day12_rtp_gate/tools/`
  - RTP-Gate / RTD 的核心实验脚本
  - dense trace 构建
  - candidate RTD 打分
  - greedy layer selection
  - lab-root 远程实验编排与续跑脚本
  - 汇总报告脚本
- `experiments/day9_small_model_repair_completion/tools/`
  - raw eval 用到的 generation / classification evaluator 包装脚本
- `experiments/day2_gemma_core_eval/tools/make_gemma_pruned_model.py`
  - 构造保存版删层模型，用于 consistency 检查
- `code/on-the-limits-of-layer-pruning/eval/gen_eval/`
  - generation eval 所需的最小辅助代码
- `reports/day12_rtp_gate/`
  - 当前导出的汇总报表快照
- `results/day12_rtp_gate/rtd_scores/`
  - 每个候选的 RTD JSON 摘要和 item-level CSV
- `tests/`
  - 一些轻量级测试，主要覆盖 lab-root 任务定义

### 4.2 仓库里不包含的内容

这个仓库**故意不包含**以下大体积或中间产物：

- 模型权重
- Hugging Face cache
- dense trace JSONL 原始文件
- raw generation/classification 的完整 JSONL 流
- 虚拟环境
- 服务器日志

这样做的原因很简单：

- 这些文件很大
- 很多只是中间产物
- 它们不适合直接跟代码和轻量结果快照一起进 GitHub

所以你在这里看到的是：

**“能帮助别人理解和复用实验方法的部分”**  
而不是  
**“把整台实验服务器完整打包搬上来”。**

---

## 5. 实验全流程概览

整个 Day12 RTP-Gate / RTD 实验，大体可以分成 5 步。

### 第 1 步：准备 dense baseline 和 dense trace

输入：

- 完整模型
- baseline JSON
- 一批标准样本（本轮主要围绕 GSM8K）

输出：

- smoke / calibration / holdout 三部分 trace
- trace manifest（记录样本数、模型路径、top-k、max tokens 等）

你可以把这一步理解成：  
**先让“没剪层的完整模型”留下标准参考轨迹。**

### 第 2 步：对候选剪层方案计算 RTD

输入：

- dense trace
- 一个候选剪层方案（例如删掉第 24 层，或者删掉 `[23,24,25]`）

输出：

- 每个候选一个 RTD JSON 摘要
- 一个 item-level CSV，用于更细地看每条样本的偏离情况

这一步是在回答：  
**“这个候选删法看起来危险不危险？”**

### 第 3 步：做 RTP-Gate greedy selection

这一步不是直接跑正式任务，而是：

- 从单层、多层候选里继续扩展
- 用 calibration split 上的 RTD 分数做 greedy 选择
- 产出类似 `rtp_gate_pure_k3`、`rtp_gate_structure_k3` 这样的候选集合

这一步回答的是：  
**“如果让我用 RTD 去挑一个更值得评测的删层组合，我会选哪组？”**

### 第 4 步：做 minimal raw evaluation

对少数关键候选，运行正式任务：

- GSM8K 500
- XSUM 500
- 7 个 classification 任务，各 200 样本

这是最终更接近“真实性能”的证据，但成本高，所以只对一小部分候选跑。

### 第 5 步：做 saved-model consistency check

选一两个典型候选：

- 真的生成删层后的模型
- 再和 runtime skip 的结果对照

这一步不是主角，但它能帮助我们避免误判。

---

## 6. 当前 lab-root 快照状态（重要：这不是最终结论）

### 6.1 这份仓库快照是什么时候导出的？

当前 GitHub 仓库里的这份结果快照，对应的说明时间是：

**2026-05-13 06:44 UTC**

请注意：

- 这是一个**进行中的实验快照**
- 不是“所有实验都结束以后”的最终归档

### 6.2 这个快照里已经明确完成的内容

从 `results/day12_rtp_gate/traces/gsm8k_dense_trace_manifest.json` 可以看到：

- `smoke = 50`
- `calibration = 200`
- `holdout = 100`
- `top_k = 100`
- `max_seq_tokens = 2048`
- `seed = 1234`

从当前导出的结果目录可以看到：

- `results/day12_rtp_gate/rtd_scores/` 中有 **237 个 RTD JSON 摘要**
- 同目录下还有 **237 个 item-level CSV**

也就是说，这个快照至少已经导出了大批候选的 RTD 打分结果。

### 6.3 当时已知的完成情况

根据这份仓库导出时保留的说明，已完成部分包括：

- dense trace manifest 构建完成
- single-layer RTD sweep 已完成
- known multi-layer baseline sweep 已完成
- RTP-Gate 候选打分已经形成大批结果快照

但当时**还没有作为最终结果写死到仓库里的**部分包括：

- RTP-Gate 最终 selection CSV
- raw GSM8K / XSUM / classification 的完整正式结果
- saved-model consistency 的最终闭环结论

所以你现在在这个仓库里看到的，更准确地说是：

**“RTP-Gate / RTD 诊断阶段已经很成形，但全套正式评测还未作为最终结果写入这个导出仓库。”**

### 6.4 当前已知最强的诊断信号

这份快照里保留的已知结论之一是：

**single-layer RTD 与旧 GSM8K 损伤参考之间，已经表现出较强一致性。**

当时记录下来的参考对照为：

- 对 Day11 saved-model GSM8K damage reference：
  - Spearman 约为 `0.805`
  - risky single-layer AUROC 约为 `0.942`
- 对 Day8 runtime-skip GSM8K damage reference：
  - Spearman 约为 `0.722`
  - AUROC 约为 `0.861`

这说明什么？

说明 RTD 至少在“识别危险层 / 危险候选”这件事上，**方向是合理的**。

### 6.5 这些对照指标应该怎么理解？

要非常注意两点：

1. 上面的相关性 / AUROC 指标主要是 **sanity check**
2. 旧服务器结果只用于参考，不是这轮 `lab-root` 的正式统计

也就是说：

- 这些数字说明 RTD 看起来不像瞎打分
- 但它们不等于“新服务器正式结论已经完工”

---

## 7. 新手快速上手：如果我想最小成本跑一次，应该怎么开始？

### 7.1 先准备什么环境？

建议你在一台 **Linux + CUDA GPU** 机器上运行。

最小前提：

- Python 3
- 可以创建 venv
- 有可用的 NVIDIA GPU
- 你已经准备好了本地模型 snapshot

### 7.2 创建环境

这一步做什么：

- 创建一个独立的 Python 虚拟环境
- 安装 RTP-Gate / RTD 需要的实验依赖

运行前需要：

- 已进入仓库根目录
- 机器上有 `python3`

成功后会得到：

- 一个可用的实验环境

```bash
python3 -m venv envs/rtp-gate
source envs/rtp-gate/bin/activate
pip install -r experiments/day12_rtp_gate/requirements-lab-root.txt
```

### 7.3 准备环境变量

这一步做什么：

- 告诉脚本仓库根目录在哪
- 告诉脚本 Hugging Face 缓存放哪
- 告诉脚本本地模型 snapshot 在哪

成功后会得到：

- 脚本能够找到模型、缓存和输出目录

```bash
export ROOT=/path/to/RTP-Gate-RTD
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/cache/huggingface/datasets"
export MODEL_PATH=/path/to/google/gemma-2-2b-it/local/snapshot
export TOKENIZERS_PARALLELISM=false
```

这些变量分别表示：

- `ROOT`：当前仓库根目录
- `HF_HOME`：Hugging Face 缓存目录
- `HF_DATASETS_CACHE`：datasets 缓存目录
- `MODEL_PATH`：本地 Gemma2-2B-It snapshot 路径
- `TOKENIZERS_PARALLELISM=false`：减少 tokenizer 并行警告和干扰

### 7.4 先跑一次 smoke

这一步做什么：

- 用很小的一批样本快速验证整条链路能不能跑通
- 确认 trace、score、JSON 输出是否正常

运行前需要：

- 环境已经装好
- `MODEL_PATH` 已配置

成功后通常会生成：

- smoke trace
- smoke candidate score
- 对应日志和状态文件

```bash
python experiments/day12_rtp_gate/tools/run_lab_root_rtp_gate_full.py \
  --root "$ROOT" \
  --model-path "$MODEL_PATH" \
  --stage smoke \
  --max-workers 1
```

如果你是第一次跑这个项目，**建议一定先跑 smoke**，因为它最省时间，也最容易暴露路径、显存、依赖和脚本参数问题。

### 7.5 再跑完整流程

这一步做什么：

- 按预设顺序执行完整的 RTP-Gate / RTD 工作流
- 包括 trace、打分、selection、consistency、raw eval、report

运行前需要：

- smoke 已经成功
- 你知道完整流程会更慢、更吃显存

成功后会得到：

- 更完整的结果目录和汇总报表

```bash
python experiments/day12_rtp_gate/tools/run_lab_root_rtp_gate_full.py \
  --root "$ROOT" \
  --model-path "$MODEL_PATH" \
  --stage all \
  --max-workers 6
```

### 7.6 lab-root 正式运行时用的配置

这份仓库导出时，正式实验使用的是：

- 项目根目录：`/root/hs/paper2_layer_pruning`
- 模型：本地 Gemma2-2B-It HF snapshot
- trace top-k：`100`
- max sequence tokens：`2048`
- GPU：`6 x RTX 2080 Ti 11GB`

这组配置对你理解“作者当时是在什么环境下跑的”很重要。

---

## 8. 结果文件怎么看？

如果你刚接触这个仓库，最值得优先看的文件如下。

### 8.1 汇总报表

- `reports/day12_rtp_gate/day12_rtp_gate_summary.json`
  - 汇总脚本生成的整体摘要
- `reports/day12_rtp_gate/day12_rtp_gate_status.md`
  - 当前状态说明
- `reports/day12_rtp_gate/rtd_scores.csv`
  - 展平后的候选 RTD 指标表
- `reports/day12_rtp_gate/top_risky_layers_by_rtd.csv`
  - 当前高风险层 / 候选视图
- `reports/day12_rtp_gate/multi_layer_comparison.csv`
  - 多层候选对比

### 8.2 逐候选结果

- `results/day12_rtp_gate/rtd_scores/*.json`
  - 每个候选一个摘要文件
- `results/day12_rtp_gate/rtd_scores/*_items.csv`
  - 每条 trace 的细粒度 item 指标

如果你想研究：

- 某个候选删了哪些层
- RTD 分数是多少
- 不同 partition 上的表现差异

那么这些文件最重要。

### 8.3 trace 配置

- `results/day12_rtp_gate/traces/gsm8k_dense_trace_manifest.json`

这个文件虽然不包含大体积 JSONL 内容，但它会告诉你：

- trace 是怎么分的
- 样本数是多少
- top-k / token 长度是多少
- 用的是哪个模型路径

### 8.4 结果目录说明

- `results/day12_rtp_gate/README.md`

这个小 README 会提醒你哪些中间产物被故意排除在仓库外面。

---

## 9. 这套方法应该怎么正确理解？它的边界在哪里？

这里非常重要，尤其是给第一次接触这类实验的人：

### 9.1 RTD 不是任务准确率

RTD 高，不等于“任务准确率一定低到某个固定数值”。

更准确的说法是：

- RTD 高，说明这个候选**更可能危险**
- RTD 低，说明这个候选**更值得继续做正式评测**

它是风险诊断，不是任务成绩单。

### 9.2 RTP-Gate 不是“全局最优剪枝算法”的宣言

这个项目的主张不是：

> “RTP-Gate 一定能找到宇宙最优剪枝方案”

而是：

> “RTP-Gate / RTD 可以作为一个实用的门控和筛选工具，先帮我们把高风险候选区分出来。”

### 9.3 XSUM / classification 在这里不是 reasoning 主证据

在本项目里：

- **GSM8K** 更像推理健康度的主证据
- **XSUM 和分类任务** 更像控制变量或负对照

它们能帮助我们看“模型是不是整体崩坏”，但不能替代数学推理主结论。

### 9.4 runtime skip 和 saved-model 不能混为一谈

如果 runtime skip 的现象很明显，但 saved-model consistency 没跟上，那么结论要更谨慎。

这也是为什么项目里专门保留了 consistency 检查这一环。

---

## 10. 常见误解 / FAQ

### Q1：这个仓库是不是完整的 Paper2 复现仓库？

不是。

它是从更大的 Paper2 工作目录中，抽出来的 **RTP-Gate / RTD 自研实验快照仓库**。

### Q2：为什么仓库里没有模型权重、HF cache、dense trace JSONL？

因为这些内容通常：

- 太大
- 太重
- 更像运行时中间产物
- 不适合作为 GitHub 代码仓库的常驻内容

### Q3：为什么要先做 dense trace，再做 RTD？

因为我们想比较的是：

**候选剪层模型在细粒度推理过程中，是怎么偏离完整模型的。**

如果没有 dense trace，就很难做这种过程级对照。

### Q4：RTD 分数高，是不是就一定不能剪？

不是“一定不能”，而是“风险更大，值得更谨慎”。

项目的目标是筛选和排序，而不是一句话封杀所有高分候选。

### Q5：为什么 README 里提到旧服务器结果？

因为旧结果可以用来做 sanity check，帮助判断 RTD 的方向是否合理。

但正式结论应该以新服务器 `lab-root` 的完整重跑为准。

### Q6：我现在能不能直接用这个仓库宣称最终实验结论？

不建议。

更准确的说法应该是：

- 这个仓库已经很好地展示了 RTP-Gate / RTD 的方法、脚本结构和中期结果快照
- 但完整 raw eval 和 consistency 的最终闭环结论，应该在后续结果 commit 里补齐

---

## 11. 最后再强调一次：这个项目的正确定位

这个项目最重要的定位不是“又一个剪枝算法名字”，而是：

**一个用于 layer pruning 风险诊断和候选门控的实验框架。**

如果你只记住一句话，希望是这句：

> RTP-Gate / RTD 想做的，是在昂贵正式评测之前，先用更便宜的信号判断哪些剪层方案可能危险，哪些方案更值得继续投入计算资源。

这也是这个仓库存在的核心价值。
