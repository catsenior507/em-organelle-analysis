# MATLAB 校验状态

[English](MATLAB-VERIFICATION-STATUS.md) ｜ **中文**

> 结论：**已完成**。`code/` 下 8 个（曾修补过的）文件全部通过
> **`mtree` 语法解析**，**语法失败数 = 0**；同时 `project_paths()` 的运行时
> 解析与 `analyze_emergent_entropy.m` 的端到端运行都已实测通过。

---

## 0. 前置：MATLAB 曾经起不来（已修复）

审计期间本机 MATLAB 完全无法启动，报
`无法与所需的 MathWorks 服务通信(错误 5201)`。这**与项目代码无关**，
是 MathWorks Service Host 锁目录的权限损坏（两个串联故障）。
根因与修复过程见 [`../tools/README-MATLAB-5201.md`](../tools/README-MATLAB-5201.md)。

修复后（普通权限、未提权令牌下）：

```
User: CAT_SISTER\user   IsAdmin: False
matlab -batch "disp(version)"
  → 25.2.0.3177638 (R2025b) Update 5
```

---

## 1. 语法校验结果

`checkcode(file,'-m0')` 在 **R2025b 上仍会返回风格/性能提示**，
所以"消息数为 0"**不是**语法是否有效的判据。真正的语法判据是
`mtree(file)`——构造语法树时会**解析**文件，语法错误直接抛异常。

`tools/check_all.m` 同时给出这两项，并把它们明确分开：

```
==== syntax validity (mtree parse) + advisory message counts ====
project_paths                syntax=OK     advisory_messages=0
project_assert_cwd           syntax=OK     advisory_messages=0
analyze_emergent_entropy     syntax=OK     advisory_messages=38
find_scale_bar               syntax=OK     advisory_messages=0
normalize_features_um        syntax=OK     advisory_messages=0
simple_region_features       syntax=OK     advisory_messages=3
extract_region_features      syntax=OK     advisory_messages=1
analyze_annotated_regions    syntax=OK     advisory_messages=5
------------------------------------------------------------------
SYNTAX FAILURES = 0
```

**8/8 语法有效，0 失败。**

### 那 38/3/1/5 条 "advisory" 是什么

全部是**风格与性能建议**，不影响运行，其中多数正是审计报告已指出的问题——
它们现在有了 MATLAB 自己的官方确认：

| 类型 | 例子 | 对应审计结论 |
|---|---|---|
| 未使用的参数/返回值 | `analyze_emergent_entropy.m` L477、L523（`estimate_lyapunov_discrete` 从未被调用）、L579、L1034 | §2.3 B5 / B8 死代码 |
| `nanmean` 建议改用 `mean` | L435、L495 | — |
| 数组在循环中增长 | L914、L933-935；`simple_region_features.m` L165/L226/L273 | 性能问题 |
| 函数已过时 | L850 `caxis`→`clim`；L1041 `datestr`/`now`→`datetime` | 兼容性 |
| 索引方式 | `analyze_annotated_regions.m` L197 | — |

**这些 advisory 我刻意没有逐条清理**：该脚本已被 `tools/network/` 取代，
它的核心问题（伪造的李雅普诺夫指数、log₂(样本数) 冒充互信息、硬编码 α→λ 查表）
不是靠消除警告能解决的，必须重写方法本身。清理警告只会让一份不该继续使用的
代码看起来更"健康"。

---

## 2. 运行时验证（不只是静态检查）

`tools/check_all.m` 同时验证 `project_paths()` 在**真实运行时**的解析结果：

```
==== project_paths() runtime resolution ====
root                : .
annotations_v2      : 1 (exists)
regions_v2          : 1 (exists)
dataset_json        : 1 (exists)
features_json       : 94 json
normalized_features : 88 csv
scale_mat           : 94 mat
python (OCR-capable): .venv\Scripts\python.exe
python probe        : status=0 out=ocr deps ok
```

### 端到端验证：路径修好后旧数字确实可复现

```bash
matlab -batch "addpath('tools'); probe_emergence"
```

`analyze_emergent_entropy.m` **首次成功运行**（生成 4 张图并写入
`data/emerg_inf_advanced/2026-10-15_20-06-21`），且数字与 2026-05-18 的旧报告
**逐项吻合**：

| 指标 | 旧报告 `2026-05-18_01-55-53` | 本次复现 |
|---|---|---|
| s_11 熵 / λ / 置信度 | 3.5368 / −0.0079 / 0.43 | **完全相同** |
| s_12 熵 / λ / 置信度 | 3.6284 / −0.0056 / 1.00 | 完全相同 |
| s_13 熵 / α / 推断 λ | 3.7502 / 0.8274 / **0.5000** | 完全相同 |
| s_18 熵 / λ / 置信度 | 3.4461 / −0.0074 / 0.67 | 完全相同 |
| s_21 熵 / α / 推断 λ | 3.5264 / 1.0657 / **0.4803** | 完全相同 |

这同时**更正了审计初版的一条错误判断**：此前写"旧输出无法由现存源码复现"，
实际原因是脚本写死了已不存在的 `d:\dachuang` 而**根本跑不起来**，
不是"跑起来了但产物对不上"。详见 [`AUDIT-REPORT.md`](AUDIT-REPORT.md) §2.3 末节。

复现也再次确认了核心问题：s_13 / s_21 的 λ 来自脚本自己打印的
`Insufficient time points for Lyapunov, using power-law...` 分支的**硬编码查表**，
而真正算出来的三个 λ 全为**负值**（按脚本自身标准即"稳定"）。

---

## 3. 结构化检查（MATLAB 不可用时的替代手段，保留作交叉验证）

```bash
python tools/check_matlab_syntax.py
```

`code/` 下 22 个 `.m` 的块关键字配对 / 跨行括号平衡 / 引号解析状态：

```
19/22 结构检查通过；3 个无法判定；0 个失败
```

3 个"无法判定"是 `analyze_annotated_regions.m`、`extract_region_features.m`、
`separate_annotation_colors.m`——纯文本解析无法区分 MATLAB 的单引号是
**转置运算符**还是**字符串起始**（如 `fieldnames(colors)'`）。
这 3 个在**改动前**的原始版本上同样"无法判定"，与本次修改无关；
现已由上面的 `mtree` 解析给出确定结论（syntax=OK）。

---

## 4. 可复现的校验命令

```bash
# 一次跑完：语法校验 + 运行时解析 + OCR 依赖探测
matlab -batch "addpath('tools'); check_all"

# 端到端：真正运行 analyze_emergent_entropy.m
# （会往 data/emerg_inf_advanced/ 写一份新的带时间戳输出）
matlab -batch "addpath('tools'); probe_emergence"
```

期望：`SYNTAX FAILURES = 0`，且 `python probe : status=0 out=ocr deps ok`。

---

## 5. 本次对 `code/` 的修改（已全部通过校验）

| 文件 | 改动 |
|---|---|
| `project_paths.m` | 🆕 按 `mfilename('fullpath')` 推算全部绝对路径，取代写死的 `D:\dachuang` 与 pwd 相对路径；附带 Python 解释器**实际 import 探测**（见下） |
| `project_assert_cwd.m` | 🆕 工作目录不符时显式报错，不再静默读错目录 |
| `analyze_emergent_entropy.m` | 路径改为 `project_paths()`（原写死 `d:\dachuang`，脚本**完全无法运行**） |
| `find_scale_bar.m` | 移除 OCR 失败后静默代入 `0.1 µm/px` 的兜底 → 改 `NaN` + `scale_ok=false` + 告警；修 `strrep(filename,'red','')` 的全局替换 |
| `normalize_features_um.m` | 每图独立 min-max → **全数据集**统计量；CSV 保留 µm 原值 |
| `simple_region_features.m` | 顶部加"数据源已废弃"守卫；`errordlg`（模态对话框，批处理下会卡死）→ `warning` |
| `extract_region_features.m` | 同上 |
| `analyze_annotated_regions.m` | 顶部加弃用说明；Python 路径改用 `project_paths()` |
| `detr_region_feature.py` | ⛔ 标注为已废弃（三条"必崩"级缺陷），指向 `tools/detr/` |

### 关于 `project_paths.m` 里的 Python 探测

本项目有**两个** Python，能力不同：

| 解释器 | 含 | 缺 |
|---|---|---|
| `.venv\Scripts\python.exe` | numpy / cv2 / PIL / pytesseract | **scipy、torch、transformers** |
| 系统 Python（PATH） | numpy / cv2 / PIL / pytesseract / **scipy** / torch / transformers | — |

`P.python` 只用于 `code\ocr.py` 与 `code\ocr_scale_bar.py`，因此探测条件是
"能 `import PIL, pytesseract`"，并**实际执行**该命令而不只是判断文件存在——
否则会出现"脚本找到了、但 import 失败"这种难查故障。
`tools\` 下的脚本（`extract_regions.py` 需要 scipy）请直接用命令行 `python`。

---

## 6. 两个写 .m 文件时的编码陷阱（本次踩到）

1. **`.m` 文件必须用系统代码页（本机 GBK）编码，或只含 ASCII。**
   我新建的调试脚本用 UTF-8 存了中文注释，MATLAB 直接报
   `文本字符无效。请检查不受支持的符号、不可见的字符或非 ASCII 字符的粘贴`
   而无法加载。项目原有的 `.m` 编码正确，所以不受影响——
   本次新增的 `tools\check_all.m`、`tools\probe_emergence.m` 因此刻意写成**纯英文**。
2. **`.bat` 文件同理**，而且更危险：cmd.exe 会把乱码**当作命令执行**。
   见 [`../tools/README-MATLAB-5201.md`](../tools/README-MATLAB-5201.md) §3。

---

## 7. 仍需注意

语法校验只证明**能被 MATLAB 解析**，不证明这些脚本**结果正确**。
`code/` 中下列文件的问题属于方法层面，已由 `tools/` 下的实现取代：

- `analyze_annotated_regions.m`：分割取的是**背景**（`~green_mask`），坐标 `[row,col]` 被当 `(x,y)` 读；
- `simple_region_features.m` / `extract_region_features.m`：解析的是**非法 JSON**（91/94 个文件含字面量 `...`）；
- `representation_learning.m` / `neural_network_training.m`：数据泄漏 + 任务病态；
- `analyze_emergent_entropy.m`：无网络重建、李雅普诺夫指数是伪造的（现已可复现，见 §2）。

请勿用这些脚本重新生成训练数据——唯一正确的数据源是
`data/annotations_v2/`（由 `tools/morpho/extract_regions.py` 生成）。
