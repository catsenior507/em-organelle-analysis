# 科研诚信问题说明

> 本文只讨论**不属于"代码 bug"范畴**的问题：那些即使代码写得完全正确，
> 也会让结论站不住脚的事项。
> 完整的代码问题清单见 [`AUDIT-REPORT.md`](AUDIT-REPORT.md)。

---

## 一、模拟生成的实验元数据被标注为"用于发表"

### 事实

`code/simulate_structure_data.py` 生成 `data/structure/structure_metadata.json`。
脚本自己的注释与字段名如下：

```python
def generate_realistic_time_series(sample_prefix, num_regions, num_time_points):
    """Generate realistic time series attributes with actual timestamps"""
    start_date = datetime(2024, 1, 1, 9, 0, 0)
    time_interval = timedelta(minutes=random.randint(5, 30))
```

产物中的 `metadata` 字段：

```json
{
  "description": "High-quality simulated structure data with comprehensive attributes",
  "generation_method": "realistic_simulation_with_statistics",
  "purpose": "Scientific research and publication",
  "version": "2.0"
}
```

即：**脚本声明这是 simulation，同时把 `purpose` 标注为用于发表。**

### 具体哪些字段不是实测值

| 字段 | 生成方式 | 是否实测 |
|---|---|---|
| `actual_date` | `datetime(2024,1,1,9,0,0) + i * timedelta(minutes=random.randint(5,30))` | ❌ |
| `timestamp` | `i + 1`（文件序号） | ❌ |
| `perturbation_level` | `random` / 线性插值到 1–7 后 `round()` | ❌ |
| `experiment_conditions_detail` | `25 + level*5 + random.gauss(0,1)`（temperature）等 | ❌ |
| `sequence_length` | 文件计数 | ❌ |
| `region_info.sampling_rate` | 同上随机的时间间隔 | ❌ |

### 这些字段如何进入分析

`code/analyze_emergent_entropy.m` 读取该文件（`:28-37`），取出
`timestamps` 与 `perturbation_levels`，然后：

- `:387-388` 用 `timestamps` 排序"时间序列"；
- `:483-484` 用 `perturbation_levels` 排序"扰动响应"；
- `:410` 的置信度公式 `min(1, num_time_points/30)` 直接由文件数决定。

因此**合成的时间戳与扰动等级参与了"李雅普诺夫指数""扰动敏感性"的计算**，
并出现在 `data/emerg_inf_advanced/*/comprehensive_report.txt` 中。

### 进度文档中的自述

`进度4.pdf` 第 3 页：

> 由于样本量有限，需要合理排布**以模拟数据**在时间序列与分类模式下不同的表征，
> 再运行脚本检验是否符合预期。

作者知道这是模拟。问题不在于"用了模拟数据做方法验证"——这在方法学上完全正当——
而在于**模拟产物与真实产物的边界在文件系统里消失了**：
同一个目录、同一套文件名、元数据里写着 purpose 是发表，
而下游脚本无法区分哪个数字来自实验、哪个来自随机数生成器。

### 处理建议（三选一，请明确选一个）

**选项 A —— 删除，只保留真实数据（推荐，如果这批图确实不是时间序列）**

```bash
python tools/data_integrity/quarantine_simulated_metadata.py --apply
```
文件被移入 `data/_QUARANTINE_simulated_metadata/` 并附 `DO_NOT_USE.md`。
然后放弃"时间序列/混沌"框架，改为跨样本统计（见第二节）。

**选项 B —— 保留，但全局改名并加显式标记**

若确实要用模拟数据验证方法（例如证明"样本量不足时李雅普诺夫指数不可估计"），
则必须：
1. 所有文件名加前缀 `SIMULATED_`；
2. 所有输出目录加 `_simulated`；
3. 所有图表标题注明"模拟数据"；
4. 论文/记录中明确写"本图基于合成数据，用于方法验证，不代表实验结果"。

**选项 C —— 换成真实元数据**

从 TIFF 的 `DateTime` 标签或仪器导出的 acquisition log 读取真实采集时间与条件。
本工作区**没有**这类文件；需要向做实验的同学索取。

---

## 二、"时间序列 / 混沌"框架缺少真实数据支撑

### 事实

`data/normalized_features/` 有 88 个 CSV。`analyze_emergent_entropy.m:53-65`
按文件名前缀（`_` 前的数字）分组，得到：

| 组 | CSV 数（= "时间点"数） |
|---|---|
| `11` | 13 |
| `12` | 49 |
| `13` | **2** |
| `18` | 20 |
| `21` | **4** |

除 11 和 12 外，各组只有 2–4 个点。而脚本自己的置信度公式
（`:410`）就写着 `min(1, num_time_points/30)` —— 即认为 30 个点才算充分。
**两组不足以做任何动力学分析**，却仍在最终排名中位列第一
（因为它们的 λ 走了 `infer_lyapunov_from_powerlaw` 的查表路径，数值偏高）。

分组依据来自文件名：

```
normalized_region_analysis__11_0002.csv
                          ^^ ^^^^
                          批  序号
```

`进度4.pdf` 第 3 页说「"_"前数字编号相同代表同一批次，于是可利用这一点**辅助添加
扰动模式**」——即"时间点"这个概念本身就是按文件名构造的，不是实验设计的时间序列。

### 影响

- "李雅普诺夫指数随时间的收敛"这类结论没有对象；
- 熵的计算把同一批次的 13–49 张图的行合并成一个矩阵（`:157-160`），
  样本量差异直接决定了熵值大小；
- 报告的排名主要由**文件数量**和**查表函数**决定，而不是生物学差异。

### 处理建议

**必须先回答一个问题**：这 88 张图是不是同一实验、按时间顺序采集的？

- **如果是**：请提供每个文件对应的真实采集时间（或扰动条件），
  并确认时间点数量；只有 2–4 个点的组必须从动力学分析中剔除。
- **如果不是**（看起来更可能）：删掉时间序列框架，改为
  **跨样本的横截面统计**——例如比较不同批次/条件下的网络拓扑指标分布。
  这不需要李雅普诺夫指数，`tools/network/` 已经按这个方向实现。

---

## 三、被引用的数字无法由现存源码复现

`data/emerg_inf_advanced/` 下有 8 次运行，其中：

- `2026-05-18_01-37-35`、`2026-05-18_01-39-03` 的报告含
  `SECTION 4: POWER-LAW ANALYSIS`、`SECTION 6: INTERPRETATION`
  以及 `Method: insufficient_levels` 等字符串，
  **现存 `analyze_emergent_entropy.m` 无法生成这些内容**；
- `2026-05-18_01-51-18` 按完整前缀分组得到 **88 个"样本"**（每张图一个），
  与同目录其他运行（5 个样本）的结论不可比；
- `data/emerg_inf/`（2026-04-15，7 次）的报告表头与单一 `unknown` 样本，
  在整个工作区里**没有任何 .m/.py 会产生它**。

也就是说，同一个分析在磁盘上留下了**三代互不兼容的结果**，
而其中的排名结论互相矛盾（01-52-49 与 01-55-53 里 s_13/s_21 都排第一，
但这两组只有 2 和 4 个时间点）。

### 处理建议

在最终报告/论文中引用任何涌现分析结果之前：
1. 明确使用哪一次运行，并在该目录写入 `PROVENANCE.md`
   （源码 commit、参数、输入文件清单）；
2. 用当前源码重跑一次，确认数字可复现；
3. 无法复现的旧输出移入 `archive/` 并从结论中剔除。

---

## 四、其它需要在记录中说明的事项

1. **OCR 从未成功识别过标注文字**。历史结果统计：504 个空字符串、
   237 个 `"M"`、36 个 `"S"`、20 个 `"co"`。
   原因是 `code/ocr.py` 未做黑底白字的极性反转。
   本次审计已确认黄色标注**全部是字形 "M"（线粒体）**——
   也就是说，**这批数据只有单一细胞器类型**，任何"多类型互作"的表述都缺少依据。
2. **`data/features_json` 中 91/94 个文件是非法 JSON**（含字面量 `...`），
   805 个区域**全部**被截断到 1000 个坐标点。
   若论文中描述的特征计算流程与实际代码不符，需要更正。
3. **比例尺标定中有 4 张图是不可信的兜底值**（0.1 µm/px，比正确值大 20–100 倍），
   另有 1 张（`12_0002`）把某个数字当成比例尺条。
   任何以 µm 为单位的结论都需要先人工核对这 5 张图。
4. **`进度3.pdf` 描述的方案与代码不一致**：文档写"匈牙利算法 + GIoU + 加权交叉熵"，
   代码里是 `torch.topk` 贪心 + MSE。记录中若描述了 DETR 的实现细节，需要与代码对齐。

---

## 五、建议的处理顺序

1. **先回答"是不是时间序列"**（第二节）——这决定第三阶段推倒还是修改；
2. **再决定模拟元数据的去留**（第一节，选项 A/B/C）；
3. 隔离工具已就绪：
   `python tools/data_integrity/quarantine_simulated_metadata.py --apply`
4. 归档 `data/emerg_inf/` 与三代混装的 `emerg_inf_advanced/`（保留但移出主目录）；
5. 用 `tools/morpho/` + `tools/network/` 重跑一条**可复现**的链路，
   并在输出目录里一起保存 `config.json` 与 git commit hash。
