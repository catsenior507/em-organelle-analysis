% 生成以微米为单位的归一化特征向量，用于机器学习处理
%
% 本次修正（对应审计结论）：
%   1. 路径改为 project_paths() 解析，不再用 pwd 相对的 '../data/...'；
%   2. 归一化统计量改为**全数据集统一**计算。旧实现按单个文件做 min-max，
%      每个文件各自拉伸到 [0,1]，样本之间的尺度差异被抹平，跨样本不可比；
%   3. CSV 导出**原始 µm 数值**（旧实现只导出 [0,1] 归一化值，把 µm 量纲丢掉了，
%      下游再也无法还原物理尺寸）；
%   4. 全局统计量（mean/std/min/max）写进每个输出 .mat，便于复现与还原。
%
% 注意：输入 data\features_all 由旧流水线（simple_region_features.m）生成，
% 其上游 features_json 多数文件不是合法 JSON 且像素被截断到 1000 点。
% 需要正确特征请使用 tools/morpho/extract_regions.py 生成的 data\annotations_v2。

% 把 code 目录加入搜索路径，并按本文件位置解析项目路径
addpath(fileparts(mfilename('fullpath')));
P = project_paths();

features_dir = P.features_all;
scale_dir = P.scale_mat;
output_dir = P.normalized_features;

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 特征列定义（顺序与旧版本一致，便于对照）
feature_names = {'width_um', 'height_um', 'center_x_um', 'center_y_um', ...
    'major_axis_um', 'minor_axis_um', 'roughness', 'complexity', ...
    'mean_gray', 'std_gray', 'aspect_ratio', 'area_um2', 'perimeter_ratio'};
feature_dim = length(feature_names);   % 13

% 获取所有 json 文件
json_files = dir(fullfile(features_dir, '*.json'));
fprintf('共发现 %d 个特征 JSON 文件\n', length(json_files));

% ==========================================================================
% 第一遍：读取全部文件的特征，汇总后计算**全数据集**的统计量
% ==========================================================================
entries = {};              % 每个元素：struct(name_part, features_um, region_ids, pixel_to_um)
all_features = zeros(0, feature_dim);
skipped_files = {};

for i = 1:length(json_files)
    % 获取当前 json 文件路径
    json_file = fullfile(features_dir, json_files(i).name);
    disp(['处理文件: ', json_file]);

    % 从 json 文件名中提取标识符（region_analysis__ 后面的部分）
    json_filename = json_files(i).name;
    [~, name_part, ~] = fileparts(json_filename);
    % 提取 region_analysis__ 后面的部分
    if contains(name_part, 'region_analysis__')
        identifier = strrep(name_part, 'region_analysis__', '');
    else
        disp('文件名格式不正确，跳过');
        continue;
    end

    % 构建对应的 mat 文件路径（scale__ 后面加上相同的标识符）
    mat_filename = ['scale__', identifier, '.mat'];
    mat_file = fullfile(scale_dir, mat_filename);

    % 检查 mat 文件是否存在
    if ~exist(mat_file, 'file')
        disp(['对应的比例尺文件不存在: ', mat_file]);
        continue;
    end

    % 读取比例尺转换比例（用 S=load(...) 而不是裸 load，避免污染工作区变量）
    S = load(mat_file);
    if ~isfield(S, 'pixel_to_um') || ~isscalar(S.pixel_to_um) || ...
            ~isfinite(S.pixel_to_um) || S.pixel_to_um <= 0
        warning('normalize_features_um:badScale', ...
            '比例尺文件缺少有效的 pixel_to_um，已跳过: %s', mat_file);
        skipped_files{end+1} = mat_file; %#ok<SAGROW>
        continue;
    end
    % find_scale_bar.m 现在会把标定失败的文件写成 scale_ok=false + pixel_to_um=NaN。
    % 标定被显式标记为不可信时**拒绝使用**，而不是像旧流程那样静默代入数值。
    if isfield(S, 'scale_ok') && ~logical(S.scale_ok)
        warning('normalize_features_um:untrustedScale', ...
            '比例尺标定被标记为不可信 (scale_ok=false)，已跳过: %s', mat_file);
        skipped_files{end+1} = mat_file; %#ok<SAGROW>
        continue;
    end
    pixel_to_um = double(S.pixel_to_um);

    % 读取 JSON 文件
    fid = fopen(json_file, 'r');
    content = fread(fid, '*char')';
    fclose(fid);

    % 解析 JSON
    data = jsondecode(content);
    regions = data.regions;

    % 提取特征向量（以微米为单位）
    num_regions = length(regions);
    features_um = zeros(num_regions, feature_dim);
    region_ids = zeros(num_regions, 1);

    for j = 1:num_regions
        region = regions(j);
        region_ids(j) = region.region_id;

        % 提取特征并转换为微米
        width = region.position.width * pixel_to_um;
        height = region.position.height * pixel_to_um;
        center_x = region.center.x * pixel_to_um;
        center_y = region.center.y * pixel_to_um;
        major_axis = region.axes.major_axis * pixel_to_um;
        minor_axis = region.axes.minor_axis * pixel_to_um;
        roughness = region.roughness; % 粗糙程度是无量纲的
        complexity = region.complexity; % 复杂程度是无量纲的
        mean_gray = region.gray_level.mean;
        std_gray = region.gray_level.std;

        % 计算额外特征
        aspect_ratio = width / max(height, 1e-9);
        area = width * height;
        perimeter_ratio = (2 * (width + height)) / max(area, 1e-9);

        % 存储特征向量
        features_um(j, :) = [width, height, center_x, center_y, major_axis, minor_axis, roughness, complexity, mean_gray, std_gray, aspect_ratio, area, perimeter_ratio];
    end

    entries{end+1} = struct('name_part', name_part, 'features_um', features_um, ...
        'region_ids', region_ids, 'pixel_to_um', pixel_to_um); %#ok<SAGROW>
    all_features = [all_features; features_um]; %#ok<AGROW>
end

if isempty(entries) || isempty(all_features)
    warning('normalize_features_um:noData', ...
        '没有可处理的区域（%d 个文件被跳过），未写出任何文件', length(skipped_files));
    return;
end

% 全数据集统计量（一次算好，所有文件共用）
global_mean = mean(all_features, 1);
global_std = std(all_features, 0, 1);
global_std(~isfinite(global_std) | global_std == 0) = 1;   % 常数特征不缩放，避免除零
global_min = min(all_features, [], 1);
global_max = max(all_features, [], 1);
global_range = global_max - global_min;
global_range(global_range == 0) = 1;

fprintf('\n全数据集统计量：%d 个文件，%d 个区域\n', length(entries), size(all_features, 1));
fprintf('归一化方式：全局 z-score = (x - global_mean) ./ global_std\n\n');

% ==========================================================================
% 第二遍：用**同一套**全局统计量写 .mat（含归一化值）与 .csv（含原始 µm 值）
% ==========================================================================
num_dataset_regions = size(all_features, 1);

for i = 1:length(entries)
    entry = entries{i};
    name_part = entry.name_part;
    features_um = entry.features_um;
    region_ids = entry.region_ids;
    pixel_to_um = entry.pixel_to_um;
    num_regions = size(features_um, 1);

    % 构建输出文件路径
    output_mat_file = fullfile(output_dir, ['normalized_', name_part, '.mat']);
    output_csv_file = fullfile(output_dir, ['normalized_', name_part, '.csv']);

    % 用全局统计量归一化（旧实现用本文件的 min/max，跨样本不可比）
    normalized_features = (features_um - global_mean) ./ global_std;

    % 处理可能的 NaN 值
    normalized_features(isnan(normalized_features)) = 0;

    % 兼容旧变量名：min_features/max_features 现在指全局范围
    min_features = global_min;
    max_features = global_max;

    % 保存为 MAT 文件（features_um 为原始 µm 值；归一化值 + 全局统计量一并保存）
    save(output_mat_file, 'normalized_features', 'features_um', 'region_ids', ...
        'global_mean', 'global_std', 'global_min', 'global_max', ...
        'min_features', 'max_features', 'feature_names', ...
        'num_dataset_regions', 'pixel_to_um');

    % 保存为 CSV 文件：导出**原始 µm 数值**，不做归一化
    % （旧实现导出 [0,1] 值却仍叫 *_um，µm 量纲被彻底丢掉）
    csv_header = ['region_id', feature_names];

    % 写入 CSV 文件
    fid = fopen(output_csv_file, 'w');
    % 写入表头
    fprintf(fid, '%s', csv_header{1});
    for k = 2:length(csv_header)
        fprintf(fid, ',%s', csv_header{k});
    end
    fprintf(fid, '\n');

    % 写入数据（原始 µm 值）
    for j = 1:num_regions
        fprintf(fid, '%d', region_ids(j));
        for k = 1:feature_dim
            fprintf(fid, ',%.6g', features_um(j, k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);

    % 显示结果
    disp(['成功处理 ', num2str(num_regions), ' 个区域']);
    disp(['转换比例: ', num2str(pixel_to_um), ' um/像素']);
    disp(['原始 µm 特征 + 全局归一化特征已保存到 ', output_mat_file]);
    disp(['CSV（原始 µm 值）已保存到 ', output_csv_file]);
    disp(['特征维度: ', num2str(feature_dim)]);
    disp('归一化范围: 全局 z-score（均值 0 / 标准差 1，统计量基于全数据集）');
    disp('--------------------------------------------------');
end

if ~isempty(skipped_files)
    warning('normalize_features_um:skipped', ...
        '共跳过 %d 个比例尺不可信的文件：\n  %s', ...
        length(skipped_files), strjoin(skipped_files, sprintf('\n  ')));
end
disp('所有文件处理完成！');
