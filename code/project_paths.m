function P = project_paths()
%PROJECT_PATHS 返回项目关键目录/文件的**绝对路径**（按本文件位置推算，与 pwd 无关）
%
%   为什么需要它
%   ------------
%   历史脚本用 `fullfile('..','data', ...)` 这类 pwd 相对路径，甚至把旧根目录
%   `D:\dachuang` 直接写死。项目被移动到 `.` 之后，
%   写死的路径直接失效（analyze_emergent_entropy.m 因此完全无法运行），相对路径
%   则会在 pwd 不对时**静默**读错目录。本函数只依赖 `mfilename('fullpath')`，
%   因此在任何工作目录下调用都能得到正确路径。
%
%   用法
%   ----
%       addpath(fileparts(mfilename('fullpath')));   % 脚本内先把自己所在目录加入搜索路径
%       P = project_paths();
%       files = dir(fullfile(P.normalized_features, '*.csv'));
%
%   数据源说明
%   ----------
%   * P.annotations_v2 / P.regions_v2 是**唯一正确**的特征数据源，由
%     `tools/morpho/extract_regions.py` 生成（1-based (x,y) 坐标，合法 JSON）。
%   * P.features_json / P.features_all 是旧流水线的产物，91/94 个文件是非法 JSON
%     且区域像素被截断到 1000 点，仅供对照，不要再用来训练模型。
%
%   输出：结构体 P，字段全部为绝对路径。

here = fileparts(mfilename('fullpath'));   % ...\dachuang\code
root = fileparts(here);                    % ...\dachuang

P = struct();
P.root = root;
P.code = here;
P.data = fullfile(root, 'data');

% —— 图像数据 ——
P.original_images  = fullfile(P.data, '原图片');            % 原始显微图
P.annotated_images = fullfile(P.data, '标注图片');          % 带红/绿/黄标注的合成图
P.separated        = fullfile(P.data, '分离的标注信息');    % 按颜色分离后的 green_/yellow_/red_

% —— 新（唯一正确）的数据源：tools/morpho/extract_regions.py 的输出 ——
P.annotations_v2    = fullfile(P.data, 'annotations_v2');
P.regions_v2        = fullfile(P.annotations_v2, 'regions');        % <image>.json
P.dataset_json      = fullfile(P.annotations_v2, 'dataset.json');   % 清单 manifest
P.scale_report_json = fullfile(P.annotations_v2, 'scale_report.json');

% —— 旧（有缺陷）的中间产物，仅作对照 ——
P.scale_mat           = fullfile(P.data, 'scale_mat');           % scale__<name>.mat
P.features_json       = fullfile(P.data, 'features_json');       % 旧区域 JSON（多数字节非法）
P.features_all        = fullfile(P.data, 'features_all');        % 旧特征汇总
P.normalized_features = fullfile(P.data, 'normalized_features'); % 归一化特征输出
P.features_pre        = fullfile(P.data, 'features_pre');
P.change_features_all = fullfile(P.data, 'change_features_all');
P.structure           = fullfile(P.data, 'structure');           % 时间/分类元数据
P.emerg_inf           = fullfile(P.data, 'emerg_inf');
P.emerg_inf_advanced  = fullfile(P.data, 'emerg_inf_advanced');  % 涌现/熵分析输出

% —— 其它 ——
P.mode_dir = fullfile(root, 'mode');   % 训练好的模型保存目录（与 code/ 同级）
P.docs     = fullfile(root, 'docs');

% —— 外部解释器与脚本（取代写死的 D:\dachuang\.venv\Scripts\python.exe）——
% 注意：本项目有**两个**可用的 Python，能力不同，别混用：
%   * .venv\Scripts\python.exe        内含 numpy/cv2/PIL/pytesseract，但**没有 scipy**
%                                     → 只适合跑 code\ocr.py、code\ocr_scale_bar.py
%   * 系统 Python（PATH 上的 python）  含 numpy/cv2/PIL/pytesseract/scipy/torch/transformers
%                                     → tools\ 下的脚本（extract_regions.py 需要 scipy）
%                                       以及 DETR 训练都必须用它
% 下面的 local_probe_python 只在**同时具备 PIL 与 pytesseract** 的候选里挑第一个，
% 保证这个字段对 code\ocr.py 一定可用；要用 tools\ 的脚本请直接用命令行 python。
P.venv_python = fullfile(root, '.venv', 'Scripts', 'python.exe');
P.python      = local_probe_python(P.venv_python);
if isempty(P.python)
    warning('project_paths:noPython', ...
        ['未找到可用的 Python（需要 PIL 与 pytesseract）。\n' ...
         '  code\\ocr.py 与 code\\ocr_scale_bar.py 将无法调用。\n' ...
         '  改用 tools\\morpho\\extract_labels.py 亦可（它自带 tesseract 路径探测）。']);
    P.python = 'python';   % 交给调用方报错，不静默兜底
end
P.ocr_script  = fullfile(here, 'ocr.py');            % analyze_annotated_regions.m 调用
P.ocr_scale_script = fullfile(here, 'ocr_scale_bar.py');  % find_scale_bar.m 调用
end


% --------------------------------------------------------------------------
function exe = local_probe_python(venv_python)
%LOCAL_PROBE_PYTHON 挑选**确实能跑 OCR 脚本**的 Python 解释器。
%
%   仅判断"文件存在"是不够的：本机 .venv 里装了 pytesseract 却没有 scipy，
%   而系统 Python 两者都有。若只按存在性挑，会出现"脚本找到了、但 import 失败"
%   这种难查的故障，所以这里实际执行一次 import 探测。
%
%   优先级：项目虚拟环境 > PATH 上的 python。
%   都不满足时返回 ''，由调用方给出明确警告——不在此处静默兜底。

% ocr.py 需要的依赖（见 code/ocr.py）。
% 探测命令刻意写成两条**直接的 import**，而不是拼
%   [importlib.import_module(m) for m in PIL,pytesseract]
% —— strjoin 产生的模块名不带引号，那会变成语法错误，
% 于是探测恒为失败、所有候选都被判为不可用（踩过这个坑）。
required = {'PIL', 'pytesseract'};
probe = ['-c "import ' strjoin(required, ', ') '"'];

candidates = { ...
    venv_python, ...                                                    % .venv\Scripts\python.exe
    fullfile(fileparts(fileparts(venv_python)), 'bin', 'python'), ...    % .venv/bin/python（类 Unix）
    fullfile(fileparts(fileparts(venv_python)), 'Scripts', 'python3.exe')};

% PATH 上的 python / python3 也作为候选
for name = {'python', 'python3'}
    [st, out] = system([name{1} ' -c "import sys;print(sys.executable)"']);
    if st == 0
        candidates{end+1} = strtrim(out); %#ok<AGROW>
    end
end

for i = 1:numel(candidates)
    cand = candidates{i};
    if isempty(cand) || ~exist(cand, 'file')
        continue;
    end
    [st, ~] = system(['"' cand '" ' probe]);
    if st == 0
        exe = cand;
        return;
    end
end

exe = '';
end
