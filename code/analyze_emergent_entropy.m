% analyze_emergent_entropy_advanced.m
% 高级涌现特征分析
% 结合香农熵与李雅普诺夫指数
% 支持时间序列和分类数据
% 使用幂律分布进行间接李雅普诺夫估计
%
% 数据结构：
% - 按前缀分组的CSV文件（相同前缀 = 相同样本，不同编号 = 时间点）
% - 每个CSV的行 = 区域，列 = 特征

clc;
clear;
close all;

% 路径统一由 project_paths() 解析：按本文件位置推算项目根目录，与 pwd 无关。
% 旧代码写死 'd:\dachuang\...'，该目录已不存在，脚本因此完全无法运行。
addpath(fileparts(mfilename('fullpath')));
P = project_paths();

fprintf('=== Advanced Emergent Entropy Analysis ===\n');
fprintf('Combining Shannon Entropy and Lyapunov Exponent Analysis\n\n');

base_path = P.normalized_features;
structure_path = P.structure;
save_path = P.emerg_inf_advanced;

time_str = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
save_dir = fullfile(save_path, time_str);
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

structure_file = fullfile(structure_path, 'structure_metadata.json');
if exist(structure_file, 'file')
    fprintf('Loading structure metadata...\n');
    structure_data = jsondecode(fileread(structure_file));
    use_structure = true;
    fprintf('Structure metadata loaded successfully.\n');
else
    fprintf('Structure metadata not found. Proceeding without temporal/classification attributes.\n');
    use_structure = false;
end

csv_files = dir(fullfile(base_path, '*.csv'));
num_files = length(csv_files);

if num_files == 0
    error('No CSV files found in normalized_features directory');
end

fprintf('Found %d CSV files for analysis.\n\n', num_files);

% 按样本前缀分组文件
sample_groups = struct();
for i = 1:num_files
    file_name = csv_files(i).name;
    [~, name, ~] = fileparts(file_name);
    parts = split(name, '__');
    if length(parts) >= 2
        full_prefix = parts{2};
        % 进一步分割以获取第一部分（例如从12_0046中获取12）
        sub_parts = split(full_prefix, '_');
        if length(sub_parts) >= 1
            sample_prefix = sub_parts{1};
        else
            sample_prefix = full_prefix;
        end
    else
        sample_prefix = 'unknown';
    end

    % 确保sample_prefix是有效的MATLAB变量名
    if ~isvarname(sample_prefix)
        sample_prefix = ['s_' sample_prefix];
    end
    
    if ~isfield(sample_groups, sample_prefix)
        sample_groups.(sample_prefix) = {};
    end
    sample_groups.(sample_prefix){end+1} = file_name;
end

sample_prefixes = fieldnames(sample_groups);
num_samples = length(sample_prefixes);
fprintf('Found %d sample groups to process.\n\n', num_samples);

% 处理每个样本组
alldata = struct();
all_entropy = struct();
all_lyapunov = struct();
all_powerlaw = struct();
all_metrics = struct();
all_structure = struct();

for s = 1:num_samples
    sample_prefix = sample_prefixes{s};
    files = sample_groups.(sample_prefix);
    num_time_points = length(files);
    
    fprintf('Processing sample %d/%d: %s (%d time points)\n', s, num_samples, sample_prefix, num_time_points);
    
    % 读取此样本的所有时间点
    time_point_data = cell(num_time_points, 1);
    timestamps = [];
    perturbation_levels = [];
    sequence_indices = [];
    all_features = {};

    for t = 1:num_time_points
        file_name = files{t};
        file_path = fullfile(base_path, file_name);
        data = readtable(file_path);

        if t == 1
            all_features = data.Properties.VariableNames(2:end);
            num_features = length(all_features);
        end

        time_point_data{t} = data;

        % 如果可用，获取结构信息
        if use_structure
            sample_keys = fieldnames(structure_data.samples);
            for j = 1:length(sample_keys)
                key = sample_keys{j};
                if contains(structure_data.samples.(key).image_file, file_name)
                    struct_info = structure_data.samples.(key);
                    if isfield(struct_info, 'timestamp')
                        timestamps(end+1) = struct_info.timestamp;
                    end
                    if isfield(struct_info, 'perturbation_level')
                        perturbation_levels(end+1) = struct_info.perturbation_level;
                    end
                    if isfield(struct_info, 'index_in_sequence')
                        sequence_indices(end+1) = struct_info.index_in_sequence;
                    end
                    break;
                end
            end
        end
    end

    % 如果可用，按序列索引排序（以确保正确的时间顺序）
    if ~isempty(sequence_indices) && length(sequence_indices) == num_time_points
        [~, sort_order] = sort(sequence_indices);
        time_point_data = time_point_data(sort_order);
        if ~isempty(timestamps)
            timestamps = timestamps(sort_order);
        end
        if ~isempty(perturbation_levels)
            perturbation_levels = perturbation_levels(sort_order);
        end
        sequence_indices = sequence_indices(sort_order);
    end

    % 跨时间点对齐区域（使用region_id）
    [aligned_feature_matrix, region_ids] = align_regions_across_time(time_point_data, all_features);
    num_regions = length(region_ids);
    num_features = length(all_features);

    % 计算香农熵（使用所有时间点的所有区域）
    all_data_matrix = [];
    for t = 1:num_time_points
        all_data_matrix = [all_data_matrix; table2array(time_point_data{t}(:, 2:end))];
    end

    entropy_results = calculate_shannon_entropy(all_data_matrix, all_features);
    all_entropy.(sample_prefix) = entropy_results;

    % 计算李雅普诺夫指数
    if num_time_points >= 5
        % 如果时间戳不可用，使用序列索引作为时间信息
        if isempty(timestamps) && ~isempty(sequence_indices)
            timestamps = sequence_indices;
        end

        if ~isempty(timestamps) && length(timestamps) == num_time_points
            fprintf('  Calculating Lyapunov exponent from time series...\n');
            lyap_results = calculate_lyapunov_time_series(aligned_feature_matrix, timestamps, num_regions, num_features);
            all_lyapunov.(sample_prefix) = lyap_results;
            all_powerlaw.(sample_prefix) = [];
        elseif ~isempty(perturbation_levels) && length(perturbation_levels) == num_time_points
            fprintf('  Calculating Lyapunov exponent from perturbation experiment...\n');
            lyap_results = calculate_lyapunov_perturbation(aligned_feature_matrix, perturbation_levels, num_regions, num_features);
            all_lyapunov.(sample_prefix) = lyap_results;
            all_powerlaw.(sample_prefix) = [];
        else
            fprintf('  Using power-law estimation for Lyapunov...\n');
            powerlaw_results = calculate_powerlaw_lyapunov(all_data_matrix, all_features);
            all_powerlaw.(sample_prefix) = powerlaw_results;
            all_lyapunov.(sample_prefix) = [];
        end
    else
        fprintf('  Insufficient time points for Lyapunov, using power-law...\n');
        powerlaw_results = calculate_powerlaw_lyapunov(all_data_matrix, all_features);
        all_powerlaw.(sample_prefix) = powerlaw_results;
        all_lyapunov.(sample_prefix) = [];
    end

    % 计算指标
    metrics = calculate_advanced_metrics(time_point_data{1}, all_data_matrix);
    all_metrics.(sample_prefix) = metrics;

    % 存储结构信息
    if use_structure
        all_structure.(sample_prefix) = struct();
        all_structure.(sample_prefix).num_time_points = num_time_points;
        if ~isempty(timestamps)
            all_structure.(sample_prefix).timestamps = timestamps;
        end
        if ~isempty(perturbation_levels)
            all_structure.(sample_prefix).perturbation_levels = perturbation_levels;
        end
    end
    
    % 打印结果
    fprintf('  Shannon Entropy: %.4f\n', entropy_results.overall_entropy);
    if ~isempty(all_lyapunov.(sample_prefix))
        fprintf('  Lyapunov Exponent (lambda): %.4f\n', all_lyapunov.(sample_prefix).lambda);
        fprintf('  Confidence: %.2f\n', all_lyapunov.(sample_prefix).confidence);
    end
    if ~isempty(all_powerlaw.(sample_prefix))
        fprintf('  Power-law Exponent (alpha): %.4f\n', all_powerlaw.(sample_prefix).mean_alpha);
        fprintf('  Inferred Lyapunov: %.4f\n', all_powerlaw.(sample_prefix).inferred_lyapunov);
    end
    fprintf('\n');
end

fprintf('Generating comprehensive analysis plots...\n\n');
generate_advanced_plots(alldata, all_entropy, all_lyapunov, all_powerlaw, all_metrics, all_structure, save_dir);

emergence_index = calculate_emergence_index(alldata, all_entropy, all_lyapunov, all_powerlaw);

generate_comprehensive_report(alldata, all_entropy, all_lyapunov, all_powerlaw, all_metrics, emergence_index, save_dir);

save(fullfile(save_dir, 'advanced_analysis_results.mat'), ...
    'alldata', 'all_entropy', 'all_lyapunov', 'all_powerlaw', 'all_metrics', 'emergence_index');

fprintf('\n=== Analysis Completed ===\n');
fprintf('Results saved to: %s\n', save_dir);


function [aligned_matrix, region_ids] = align_regions_across_time(time_point_data, feature_names)
    num_time_points = length(time_point_data);
    num_features = length(feature_names);

    % 收集所有唯一的区域ID
    all_region_ids = [];
    for t = 1:num_time_points
        all_region_ids = [all_region_ids; time_point_data{t}.region_id];
    end
    region_ids = unique(all_region_ids);
    num_regions = length(region_ids);

    % 创建对齐矩阵：[num_regions x num_features x num_time_points]
    aligned_matrix = NaN(num_regions, num_features, num_time_points);

    for t = 1:num_time_points
        data = time_point_data{t};
        [~, idx] = ismember(data.region_id, region_ids);
        aligned_matrix(idx, :, t) = table2array(data(:, 2:end));
    end
end


function entropy_results = calculate_shannon_entropy(feature_matrix, feature_cols)
    num_features = size(feature_matrix, 2);
    num_regions = size(feature_matrix, 1);

    entropy_results = struct();

    feature_entropy = zeros(1, num_features);
    for j = 1:num_features
        feature_data = feature_matrix(:, j);
        feature_entropy(j) = compute_shannon_entropy(feature_data);
    end
    entropy_results.feature_entropy = feature_entropy;
    entropy_results.feature_names = feature_cols;

    entropy_results.overall_entropy = compute_shannon_entropy(feature_matrix(:));

    entropy_results.joint_entropy = compute_joint_entropy(feature_matrix);

    if any(strcmp(feature_cols, 'center_x_um')) && any(strcmp(feature_cols, 'center_y_um'))
        x_idx = find(strcmp(feature_cols, 'center_x_um'));
        y_idx = find(strcmp(feature_cols, 'center_y_um'));
        spatial_data = [feature_matrix(:, x_idx), feature_matrix(:, y_idx)];
        entropy_results.spatial_entropy = compute_shannon_entropy(spatial_data(:));
    end

    entropy_results.conditional_entropy = compute_conditional_entropy(feature_matrix);

    entropy_results.mutual_info = entropy_results.joint_entropy - entropy_results.conditional_entropy;
end


function entropy = compute_shannon_entropy(data, num_bins)
    if nargin < 2
        num_bins = 20;
    end

    valid_data = data(~isnan(data));
    if length(unique(valid_data)) <= 1
        entropy = 0;
        return;
    end

    [counts, ~] = histcounts(valid_data, num_bins);
    counts = counts(counts > 0);
    prob = counts / sum(counts);

    prob = prob(prob > 0);
    entropy = -sum(prob .* log2(prob));
end


function joint_entropy = compute_joint_entropy(data_matrix)
    num_features = size(data_matrix, 2);

    if num_features < 2
        joint_entropy = compute_shannon_entropy(data_matrix(:));
        return;
    end

    data_valid = data_matrix(~any(isnan(data_matrix), 2), :);
    if isempty(data_valid)
        joint_entropy = 0;
        return;
    end

    num_bins_per_feature = 5;
    bin_indices = zeros(size(data_valid, 1), num_features);
    for j = 1:num_features
        feat = data_valid(:, j);
        edges = linspace(min(feat) - 0.01, max(feat) + 0.01, num_bins_per_feature + 1);
        bin_indices(:, j) = discretize(feat, edges);
    end

    combined_bins = zeros(size(bin_indices, 1), 1);
    bin_multipliers = num_bins_per_feature.^(0:num_features-1);
    for i = 1:size(bin_indices, 1)
        combined_bins(i) = dot(bin_indices(i, :), bin_multipliers);
    end

    unique_bins = unique(combined_bins);
    counts = histcounts(combined_bins, [unique_bins; max(unique_bins) + 1]);
    counts = counts(counts > 0);
    prob = counts / sum(counts);
    prob = prob(prob > 0);

    joint_entropy = -sum(prob .* log2(prob));
end


function cond_entropy = compute_conditional_entropy(data_matrix)
    num_features = size(data_matrix, 2);

    if num_features < 2
        cond_entropy = 0;
        return;
    end

    cond_entropy = 0;
    for j = 1:num_features
        other_features = [1:j-1, j+1:num_features];
        sub_matrix = data_matrix(:, other_features);
        cond_entropy = cond_entropy + compute_conditional_entropy_single(data_matrix(:, j), sub_matrix);
    end
    cond_entropy = cond_entropy / num_features;
end


function ce = compute_conditional_entropy_single(target, features)
    combined = [features, target];
    joint_h = compute_joint_entropy(combined);
    features_h = compute_joint_entropy(features);
    ce = joint_h - features_h;
end


function lyap_results = calculate_lyapunov_time_series(aligned_matrix, timestamps, num_regions, num_features)
    lyap_results = struct();
    num_time_points = size(aligned_matrix, 3);

    if num_time_points < 5
        lyap_results.lambda = NaN;
        lyap_results.confidence = 0;
        lyap_results.method = 'insufficient_data';
        return;
    end

    [~, sorted_idx] = sort(timestamps);
    sorted_data = aligned_matrix(:, :, sorted_idx);

    % 使用多尺度离散化计算特征-wise李雅普诺夫指数
    lambda_values = zeros(1, num_features);
    valid_features = false(1, num_features);

    for j = 1:num_features
        feature_time_series = squeeze(sorted_data(:, j, :));

        % 尝试多个离散化级别
        for num_levels = [3, 4, 5, 6, 7]
            [lambda_j, success] = estimate_lyapunov_for_feature(feature_time_series, num_levels);
            if success
                lambda_values(j) = lambda_j;
                valid_features(j) = true;
                break;
            end
        end
    end

    if any(valid_features)
        lyap_results.lambda = mean(lambda_values(valid_features));
        lyap_results.confidence = min(1, num_time_points / 30);
    else
        lyap_results.lambda = 0;
        lyap_results.confidence = 0.1;
    end

    lyap_results.method = 'multi_scale_discrete';
    lyap_results.num_time_points = num_time_points;
    lyap_results.feature_lambdas = lambda_values;

    perturbations = compute_perturbation_sensitivity(sorted_data);
    lyap_results.avg_sensitivity = mean(perturbations);
    lyap_results.max_sensitivity = max(perturbations);
end


function [lambda, success] = estimate_lyapunov_for_feature(feature_ts, num_levels)
    [num_regions, num_time_points] = size(feature_ts);

    % 使用跨区域的均值作为系统状态
    state_ts = nanmean(feature_ts, 1);
    valid_states = ~isnan(state_ts);
    state_ts = state_ts(valid_states);

    if length(state_ts) < 5
        lambda = 0;
        success = false;
        return;
    end

    % 离散化状态时间序列
    q = quantile(state_ts, linspace(0, 1, num_levels + 1));
    discretized = discretize(state_ts, q);

    % 计算发散度
    divergences = zeros(length(state_ts) - 1, 1);
    for t = 1:length(state_ts) - 1
        divergences(t) = abs(discretized(t + 1) - discretized(t));
    end

    valid_divs = divergences(divergences > 0);
    if length(valid_divs) < 3
        lambda = 0;
        success = false;
        return;
    end

    % 从发散度估计指数
    log_divs = log(valid_divs + 1e-10);
    t_vector = (1:length(log_divs))';

    if length(t_vector) > 3
        coeffs = polyfit(t_vector, log_divs, 1);
        lambda = coeffs(1);
        success = true;
    else
        lambda = 0;
        success = false;
    end
end


function lyap_results = calculate_lyapunov_perturbation(aligned_matrix, perturbation_levels, num_regions, num_features)
    lyap_results = struct();
    num_time_points = size(aligned_matrix, 3);

    if num_time_points < 3
        lyap_results.lambda = NaN;
        lyap_results.confidence = 0;
        lyap_results.method = 'insufficient_levels';
        return;
    end

    [~, sorted_idx] = sort(perturbation_levels);
    sorted_data = aligned_matrix(:, :, sorted_idx);

    % 从扰动响应估计李雅普诺夫指数
    lambda_values = zeros(1, num_features);
    for j = 1:num_features
        feature_series = squeeze(sorted_data(:, j, :));
        mean_feature = nanmean(feature_series, 1);

        valid_idx = ~isnan(mean_feature);
        mean_feature = mean_feature(valid_idx);

        if length(mean_feature) >= 3
            diffs = diff(mean_feature);
            if any(abs(diffs) > 0.01)
                lambda_values(j) = mean(abs(diffs));
            end
        end
    end

    lyap_results.lambda = mean(lambda_values(lambda_values > 0));
    if isnan(lyap_results.lambda)
        lyap_results.lambda = 0;
    end

    lyap_results.confidence = min(1, num_time_points / 15);
    lyap_results.method = 'perturbation_response';
    lyap_results.num_levels = num_time_points;

    perturbations = compute_perturbation_sensitivity(sorted_data);
    lyap_results.avg_sensitivity = mean(perturbations);
    lyap_results.max_sensitivity = max(perturbations);
end


function lambda = estimate_lyapunov_discrete(discretized_data)
    [T, N] = size(discretized_data);

    if T < 5 || N < 1
        lambda = NaN;
        return;
    end

    divergences = zeros(T - 1, 1);
    for t = 1:T - 1
        state_diff = abs(discretized_data(t + 1, :) - discretized_data(t, :));
        divergences(t) = mean(state_diff);
    end

    valid_divs = divergences(divergences > 0);
    if isempty(valid_divs) || length(valid_divs) < 2
        lambda = 0;
        return;
    end

    valid_divs = valid_divs(valid_divs < prctile(valid_divs, 95));

    t_vector = (1:length(valid_divs))';
    log_divs = log(valid_divs + 1e-10);

    if length(t_vector) > 2
        coeffs = polyfit(t_vector, log_divs, 1);
        lambda = coeffs(1);
    else
        lambda = 0;
    end
end


function perturbations = compute_perturbation_sensitivity(data)
    num_time_points = size(data, 3);
    num_features = size(data, 2);

    perturbations = zeros(num_time_points - 1, 1);
    for t = 1:num_time_points - 1
        original = squeeze(data(:, :, t));
        perturbed = squeeze(data(:, :, t + 1));

        valid_idx = ~isnan(original(:)) & ~isnan(perturbed(:));
        if any(valid_idx)
            rel_diff = abs(perturbed(valid_idx) - original(valid_idx)) ./ (abs(original(valid_idx)) + 1e-10);
            perturbations(t) = mean(rel_diff);
        else
            perturbations(t) = NaN;
        end
    end

    perturbations = perturbations(~isnan(perturbations));
end


function powerlaw_results = calculate_powerlaw_lyapunov(feature_matrix, feature_cols)
    powerlaw_results = struct();

    num_features = size(feature_matrix, 2);
    num_regions = size(feature_matrix, 1);

    alpha_values = zeros(1, num_features);
    r_squared = zeros(1, num_features);

    for j = 1:num_features
        feature_data = feature_matrix(:, j);
        feature_data = feature_data(feature_data > 0);

        if length(feature_data) < 10
            alpha_values(j) = NaN;
            r_squared(j) = 0;
            continue;
        end

        sorted_data = sort(feature_data, 'descend');
        ranks = (1:length(sorted_data))';

        log_ranks = log(ranks);
        log_data = log(sorted_data);

        valid_idx = ~isinf(log_ranks) & ~isinf(log_data);
        log_ranks = log_ranks(valid_idx);
        log_data = log_data(valid_idx);

        if length(log_ranks) > 5
            coeffs = polyfit(log_ranks, log_data, 1);
            alpha_values(j) = -coeffs(1);

            y_hat = polyval(coeffs, log_ranks);
            residuals = log_data - y_hat;
            ss_res = sum(residuals.^2);
            ss_tot = sum((log_data - mean(log_data)).^2);
            r_squared(j) = 1 - ss_res / (ss_tot + eps);
        else
            alpha_values(j) = NaN;
            r_squared(j) = 0;
        end
    end

    powerlaw_results.alpha = alpha_values;
    powerlaw_results.r_squared = r_squared;

    valid_alpha = alpha_values(~isnan(alpha_values) & r_squared > 0.3);
    if isempty(valid_alpha)
        powerlaw_results.mean_alpha = 1.5;
    else
        powerlaw_results.mean_alpha = mean(valid_alpha);
    end

    powerlaw_results.inferred_lyapunov = infer_lyapunov_from_powerlaw(powerlaw_results.mean_alpha, num_regions);
end


function inferred_lambda = infer_lyapunov_from_powerlaw(alpha, N)
    if alpha < 1
        inferred_lambda = 0.5;
    elseif alpha < 2
        inferred_lambda = 0.2 + (2 - alpha) * 0.3;
    elseif alpha < 3
        inferred_lambda = 0.1 + (3 - alpha) * 0.1;
    else
        inferred_lambda = 0.05;
    end

    if N > 100
        inferred_lambda = inferred_lambda * 1.2;
    elseif N < 10
        inferred_lambda = inferred_lambda * 0.8;
    end
end


function metrics = calculate_advanced_metrics(sample_data, all_data)
    metrics = struct();

    metrics.num_regions = height(sample_data);

    correlation_matrix = corrcoef(all_data, 'Rows', 'complete');
    if size(correlation_matrix, 1) > 1
        metrics.correlation_matrix = correlation_matrix;
    else
        metrics.correlation_matrix = 1;
    end

    metrics.mean_feature = mean(all_data, 1, 'omitnan');
    metrics.std_feature = std(all_data, 0, 1, 'omitnan');
end


function emergence_index = calculate_emergence_index(alldata, all_entropy, all_lyapunov, all_powerlaw)
    sample_names = fieldnames(all_entropy);
    num_samples = length(sample_names);

    emergence_index = struct();
    emergence_index.shannon_entropy = zeros(num_samples, 1);
    emergence_index.lyapunov_estimate = zeros(num_samples, 1);
    emergence_index.combined_index = zeros(num_samples, 1);
    emergence_index.sample_names = sample_names;

    for i = 1:num_samples
        sample_name = sample_names{i};
        emergence_index.shannon_entropy(i) = all_entropy.(sample_name).overall_entropy;

        if ~isempty(all_lyapunov.(sample_name))
            emergence_index.lyapunov_estimate(i) = all_lyapunov.(sample_name).lambda;
        elseif ~isempty(all_powerlaw.(sample_name))
            emergence_index.lyapunov_estimate(i) = all_powerlaw.(sample_name).inferred_lyapunov;
        else
            emergence_index.lyapunov_estimate(i) = NaN;
        end

        if ~isnan(emergence_index.lyapunov_estimate(i))
            emergence_index.combined_index(i) = ...
                emergence_index.shannon_entropy(i) * (1 + abs(emergence_index.lyapunov_estimate(i)));
        else
            emergence_index.combined_index(i) = emergence_index.shannon_entropy(i);
        end
    end

    emergence_index.mean_shannon = mean(emergence_index.shannon_entropy);
    emergence_index.std_shannon = std(emergence_index.shannon_entropy);

    valid_lyap = emergence_index.lyapunov_estimate(~isnan(emergence_index.lyapunov_estimate));
    if ~isempty(valid_lyap)
        emergence_index.mean_lyapunov = mean(valid_lyap);
        emergence_index.std_lyapunov = std(valid_lyap);
    else
        emergence_index.mean_lyapunov = NaN;
        emergence_index.std_lyapunov = NaN;
    end

    emergence_index.samples = struct();
    for i = 1:num_samples
        emergence_index.samples.(sample_names{i}) = struct();
        emergence_index.samples.(sample_names{i}).shannon = emergence_index.shannon_entropy(i);
        emergence_index.samples.(sample_names{i}).lyapunov_estimate = emergence_index.lyapunov_estimate(i);
        emergence_index.samples.(sample_names{i}).combined = emergence_index.combined_index(i);
    end
end


function generate_advanced_plots(alldata, all_entropy, all_lyapunov, all_powerlaw, all_metrics, all_structure, save_dir)
    % 设置出版质量的图形样式
    set(groot, 'defaultAxesFontSize', 12);
    set(groot, 'defaultAxesFontWeight', 'normal');
    set(groot, 'defaultLineLineWidth', 2);
    set(groot, 'defaultAxesLineWidth', 1.5);
    set(groot, 'defaultFigureColor', 'w');

    sample_names = fieldnames(all_entropy);
    num_samples = length(sample_names);

    shannon_vals = zeros(num_samples, 1);
    lyapunov_vals = zeros(num_samples, 1);
    lyapunov_conf = zeros(num_samples, 1);
    combined_vals = zeros(num_samples, 1);
    lyap_method = cell(num_samples, 1);

    for i = 1:num_samples
        shannon_vals(i) = all_entropy.(sample_names{i}).overall_entropy;

        if ~isempty(all_lyapunov.(sample_names{i}))
            lyapunov_vals(i) = all_lyapunov.(sample_names{i}).lambda;
            lyapunov_conf(i) = all_lyapunov.(sample_names{i}).confidence;
            if isfield(all_lyapunov.(sample_names{i}), 'method')
                lyap_method{i} = all_lyapunov.(sample_names{i}).method;
            else
                lyap_method{i} = 'direct';
            end
        elseif ~isempty(all_powerlaw.(sample_names{i}))
            lyapunov_vals(i) = all_powerlaw.(sample_names{i}).inferred_lyapunov;
            lyapunov_conf(i) = 0.5;
            lyap_method{i} = 'power-law';
        else
            lyapunov_vals(i) = NaN;
            lyapunov_conf(i) = 0;
            lyap_method{i} = 'unknown';
        end

        if ~isnan(lyapunov_vals(i))
            combined_vals(i) = shannon_vals(i) * (1 + abs(lyapunov_vals(i)));
        else
            combined_vals(i) = shannon_vals(i);
        end
    end

    % 定义专业配色方案
    colors = [
        31, 119, 180;   % 蓝色
        255, 127, 14;   % 橙色
        44, 160, 44;    % 绿色
        214, 39, 40;    % 红色
        148, 103, 189   % 紫色
    ] / 255;

    % ============================================================
    % 图1：综合指标概览（3个子图）
    % ============================================================
    fig1 = figure('Position', [50, 50, 1400, 500], 'PaperPositionMode', 'auto');

    % 子图1：香农熵
    ax1 = subplot(1, 3, 1);
    bars1 = bar(1:num_samples, shannon_vals, 'FaceColor', colors(1, :), 'EdgeColor', 'k', 'LineWidth', 1.2);
    set(gca, 'XTickLabel', sample_names, 'XTickLabelRotation', 45);
    xlabel('Sample ID', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Entropy (bits)', 'FontSize', 13, 'FontWeight', 'bold');
    title('(a) Shannon Entropy', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    box on;
    set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);
    
    % Subplot 2: Lyapunov Exponent with confidence
    ax2 = subplot(1, 3, 2);
    bars2 = bar(1:num_samples, lyapunov_vals, 'FaceColor', colors(2, :), 'EdgeColor', 'k', 'LineWidth', 1.2);
    set(gca, 'XTickLabel', sample_names, 'XTickLabelRotation', 45);
    xlabel('Sample ID', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Lyapunov Exponent (\lambda)', 'FontSize', 13, 'FontWeight', 'bold');
    title('(b) Lyapunov Exponent', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    box on;
    set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);

    % 添加置信区间标记
    hold on;
    for i = 1:num_samples
        if ~isnan(lyapunov_vals(i))
            errorbar(i, lyapunov_vals(i), lyapunov_vals(i)*0.1*lyapunov_conf(i), ...
                'k.', 'LineStyle', 'none', 'MarkerSize', 10);
        end
    end
    hold off;

    % 子图3：综合涌现指数
    ax3 = subplot(1, 3, 3);
    bars3 = bar(1:num_samples, combined_vals, 'FaceColor', colors(3, :), 'EdgeColor', 'k', 'LineWidth', 1.2);
    set(gca, 'XTickLabel', sample_names, 'XTickLabelRotation', 45);
    xlabel('Sample ID', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Emergence Index', 'FontSize', 13, 'FontWeight', 'bold');
    title('(c) Combined Emergence Index', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    box on;
    set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);

    % 链接坐标轴以便更好地可视化
    linkaxes([ax1, ax2, ax3], 'x');

    % 以多种格式保存
    print(fig1, fullfile(save_dir, 'combined_metrics_figure1'), '-dpng', '-r300');
    print(fig1, fullfile(save_dir, 'combined_metrics_figure1'), '-depsc');
    savefig(fig1, fullfile(save_dir, 'combined_metrics_figure1.fig'));
    close(fig1);

    % ============================================================
    % 图2：熵 vs 李雅普诺夫散点图
    % ============================================================
    fig2 = figure('Position', [100, 100, 900, 700], 'PaperPositionMode', 'auto');

    % 创建基于置信度大小的散点图
    marker_sizes = max(80, lyapunov_conf * 200);
    [~, sorted_idx] = sort(combined_vals, 'descend');

    scatter_obj = scatter(shannon_vals, lyapunov_vals, marker_sizes, combined_vals, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);

    colorbar_obj = colorbar;
    ylabel(colorbar_obj, 'Emergence Index', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(flipud(hot));
    caxis([min(combined_vals), max(combined_vals)]);
    
    xlabel('Shannon Entropy (bits)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Lyapunov Exponent (\lambda)', 'FontSize', 14, 'FontWeight', 'bold');
    title('Relationship Between Shannon Entropy and Lyapunov Exponent', 'FontSize', 15, 'FontWeight', 'bold');
    grid on; grid minor;
    box on;
    set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.15);

    % 添加样本标签
    hold on;
    for i = 1:num_samples
        text(shannon_vals(i) + 0.05*range(shannon_vals), ...
             lyapunov_vals(i) + 0.05*range(lyapunov_vals(~isnan(lyapunov_vals))), ...
             sample_names{i}, 'FontSize', 9, 'FontWeight', 'bold');
    end
    hold off;

    % 添加趋势线
    valid_idx = ~isnan(lyapunov_vals);
    if sum(valid_idx) > 1
        p = polyfit(shannon_vals(valid_idx), lyapunov_vals(valid_idx), 1);
        x_trend = linspace(min(shannon_vals), max(shannon_vals), 100);
        y_trend = polyval(p, x_trend);
        hold on;
        plot(x_trend, y_trend, '--', 'Color', colors(4, :), 'LineWidth', 2);
        hold off;
    end

    % 保存
    print(fig2, fullfile(save_dir, 'entropy_vs_lyapunov_figure2'), '-dpng', '-r300');
    print(fig2, fullfile(save_dir, 'entropy_vs_lyapunov_figure2'), '-depsc');
    savefig(fig2, fullfile(save_dir, 'entropy_vs_lyapunov_figure2.fig'));
    close(fig2);

    % ============================================================
    % 图3：特征-wise李雅普诺夫分析（如果可用）
    % ============================================================
    has_lyap_features = false;
    for i = 1:num_samples
        if ~isempty(all_lyapunov.(sample_names{i})) && isfield(all_lyapunov.(sample_names{i}), 'feature_lambda')
            has_lyap_features = true;
            break;
        end
    end

    if has_lyap_features
        fig3 = figure('Position', [100, 100, 1200, 600], 'PaperPositionMode', 'auto');

        % 从第一个样本获取特征数量
        feat_count = 0;
        for i = 1:num_samples
            if ~isempty(all_lyapunov.(sample_names{i})) && isfield(all_lyapunov.(sample_names{i}), 'feature_lambda')
                feat_count = length(all_lyapunov.(sample_names{i}).feature_lambda);
                break;
            end
        end

        if feat_count > 0
            subplot(1, 2, 1);
            % 特征-wise李雅普诺夫指数的箱线图
            feature_lyaps = [];
            for i = 1:num_samples
                if ~isempty(all_lyapunov.(sample_names{i})) && isfield(all_lyapunov.(sample_names{i}), 'feature_lambda')
                    feature_lyaps = [feature_lyaps; all_lyapunov.(sample_names{i}).feature_lambda];
                end
            end

            if ~isempty(feature_lyaps)
                boxplot(feature_lyaps, 'Colors', colors(1, :), 'Widths', 0.6);
                xlabel('Feature Index', 'FontSize', 13, 'FontWeight', 'bold');
                ylabel('Feature-wise Lyapunov Exponent', 'FontSize', 13, 'FontWeight', 'bold');
                title('(a) Feature-wise Lyapunov Distribution', 'FontSize', 14, 'FontWeight', 'bold');
                grid on; box on;
            end

            subplot(1, 2, 2);
            % 如果可用，幂律比较
            all_alpha_vals = [];
            all_shannon_vals = [];
            sample_labels = {};
            for i = 1:num_samples
                if ~isempty(all_powerlaw.(sample_names{i}))
                    all_alpha_vals = [all_alpha_vals; all_powerlaw.(sample_names{i}).mean_alpha];
                    all_shannon_vals = [all_shannon_vals; shannon_vals(i)];
                    sample_labels = [sample_labels; sample_names(i)];
                end
            end

            if length(all_alpha_vals) > 1
                scatter(all_shannon_vals, all_alpha_vals, 150, colors(2, :), 'filled', 'MarkerEdgeColor', 'k');
                xlabel('Shannon Entropy (bits)', 'FontSize', 13, 'FontWeight', 'bold');
                ylabel('Power-law Exponent (\alpha)', 'FontSize', 13, 'FontWeight', 'bold');
                title('(b) Entropy vs Power-law Exponent', 'FontSize', 14, 'FontWeight', 'bold');
                grid on; grid minor; box on;

                % 添加标签
                hold on;
                for i = 1:length(all_alpha_vals)
                    text(all_shannon_vals(i) + 0.03*range(all_shannon_vals), ...
                         all_alpha_vals(i) + 0.03*range(all_alpha_vals), ...
                         sample_labels{i}, 'FontSize', 10, 'FontWeight', 'bold');
                end
                hold off;

                % 添加趋势线
                p_alpha = polyfit(all_shannon_vals, all_alpha_vals, 1);
                x_trend_a = linspace(min(all_shannon_vals), max(all_shannon_vals), 100);
                y_trend_a = polyval(p_alpha, x_trend_a);
                hold on;
                plot(x_trend_a, y_trend_a, '--', 'Color', colors(3, :), 'LineWidth', 2);
                hold off;
            end
        end

        % 保存
        print(fig3, fullfile(save_dir, 'feature_analysis_figure3'), '-dpng', '-r300');
        print(fig3, fullfile(save_dir, 'feature_analysis_figure3'), '-depsc');
        savefig(fig3, fullfile(save_dir, 'feature_analysis_figure3.fig'));
        close(fig3);
    end

    % ============================================================
    % 图4：汇总统计（用于大样本）
    % ============================================================
    if num_samples >= 3
        fig4 = figure('Position', [100, 100, 1000, 800], 'PaperPositionMode', 'auto');

        % 2x2子图网格
        subplot(2, 2, 1);
        histogram(shannon_vals, 'FaceColor', colors(1, :), 'EdgeColor', 'w', 'LineWidth', 1.5);
        xlabel('Shannon Entropy (bits)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Count', 'FontSize', 12, 'FontWeight', 'bold');
        title('(a) Entropy Distribution', 'FontSize', 13, 'FontWeight', 'bold');
        grid on; box on;

        subplot(2, 2, 2);
        valid_lyap = lyapunov_vals(~isnan(lyapunov_vals));
        if length(valid_lyap) >= 2
            histogram(valid_lyap, 'FaceColor', colors(2, :), 'EdgeColor', 'w', 'LineWidth', 1.5);
            xlabel('Lyapunov Exponent (\lambda)', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Count', 'FontSize', 12, 'FontWeight', 'bold');
            title('(b) Lyapunov Distribution', 'FontSize', 13, 'FontWeight', 'bold');
            grid on; box on;
        end

        subplot(2, 2, 3);
        [~, sort_idx] = sort(combined_vals, 'descend');
        sorted_combined = combined_vals(sort_idx);
        sorted_names = sample_names(sort_idx);
        bar(1:num_samples, sorted_combined, 'FaceColor', colors(3, :), 'EdgeColor', 'k', 'LineWidth', 1.2);
        set(gca, 'XTickLabel', sorted_names, 'XTickLabelRotation', 45);
        xlabel('Sample ID (Ranked)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Emergence Index', 'FontSize', 12, 'FontWeight', 'bold');
        title('(c) Emergence Index Ranking', 'FontSize', 13, 'FontWeight', 'bold');
        grid on; box on;

        subplot(2, 2, 4);
        % 雷达/蛛网图或小提琴图的替代方案
        metrics_matrix = [shannon_vals, abs(lyapunov_vals), combined_vals];
        metrics_matrix = metrics_matrix ./ max(metrics_matrix, [], 1);  % Normalize
        
        metrics_labels = {'Entropy', '|Lyapunov|', 'Emergence'};
        [X, Y] = meshgrid(1:3, 1:num_samples);
        surf(X, Y, metrics_matrix, 'FaceAlpha', 0.7, 'EdgeColor', 'none');
        colormap(parula);
        colorbar;
        set(gca, 'XTick', 1:3, 'XTickLabel', metrics_labels);
        ylabel('Sample Index', 'FontSize', 12, 'FontWeight', 'bold');
        title('(d) Normalized Metrics Overview', 'FontSize', 13, 'FontWeight', 'bold');
        view(30, 45);
        box on;

        % 保存
        print(fig4, fullfile(save_dir, 'summary_statistics_figure4'), '-dpng', '-r300');
        print(fig4, fullfile(save_dir, 'summary_statistics_figure4'), '-depsc');
        savefig(fig4, fullfile(save_dir, 'summary_statistics_figure4.fig'));
        close(fig4);
    end
    
    fprintf('  Generated %d publication-quality plots (PNG + EPS + FIG formats).\n', 3 + (num_samples >=3));
end


function generate_comprehensive_report(alldata, all_entropy, all_lyapunov, all_powerlaw, all_metrics, emergence_index, save_dir)
    fid = fopen(fullfile(save_dir, 'comprehensive_report.txt'), 'w');
    
    fprintf(fid, '============================================================\n');
    fprintf(fid, '    ADVANCED EMERGENT ENTROPY ANALYSIS REPORT\n');
    fprintf(fid, '    Comprehensive Shannon Entropy + Lyapunov Exponent\n');
    fprintf(fid, '============================================================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', datestr(now));
    fprintf(fid, '============================================================\n\n');
    
    sample_names = fieldnames(all_entropy);
    num_samples = length(sample_names);
    
    fprintf(fid, 'SECTION 1: DATASET OVERVIEW\n');
    fprintf(fid, '------------------------------------------------------------\n');
    fprintf(fid, 'Total samples analyzed: %d\n\n', num_samples);
    
    for i = 1:num_samples
        sample_name = sample_names{i};
        fprintf(fid, 'Sample %d: %s\n', i, sample_name);
        if isfield(all_metrics.(sample_name), 'num_regions')
            fprintf(fid, '  Regions (first image): %d\n', all_metrics.(sample_name).num_regions);
        end
        if isfield(all_entropy.(sample_name), 'feature_names')
            fprintf(fid, '  Features: %d\n', length(all_entropy.(sample_name).feature_names));
        end
    end
    fprintf(fid, '\n');
    
    fprintf(fid, 'SECTION 2: SHANNON ENTROPY ANALYSIS\n');
    fprintf(fid, '------------------------------------------------------------\n');
    fprintf(fid, 'Overall Statistics:\n');
    fprintf(fid, '  Mean Shannon Entropy: %.4f (std: %.4f)\n', emergence_index.mean_shannon, emergence_index.std_shannon);
    fprintf(fid, '  Min Shannon Entropy: %.4f\n', min(emergence_index.shannon_entropy));
    fprintf(fid, '  Max Shannon Entropy: %.4f\n', max(emergence_index.shannon_entropy));
    fprintf(fid, '\n');
    
    for i = 1:num_samples
        sample_name = sample_names{i};
        fprintf(fid, 'Sample %s:\n', sample_name);
        fprintf(fid, '  Overall Entropy: %.4f\n', all_entropy.(sample_name).overall_entropy);
        if isfield(all_entropy.(sample_name), 'spatial_entropy')
            fprintf(fid, '  Spatial Entropy: %.4f\n', all_entropy.(sample_name).spatial_entropy);
        end
        if isfield(all_entropy.(sample_name), 'joint_entropy')
            fprintf(fid, '  Joint Entropy: %.4f\n', all_entropy.(sample_name).joint_entropy);
        end
        if isfield(all_entropy.(sample_name), 'mutual_info')
            fprintf(fid, '  Mutual Information: %.4f\n', all_entropy.(sample_name).mutual_info);
        end
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'SECTION 3: LYAPUNOV EXPONENT ANALYSIS\n');
    fprintf(fid, '------------------------------------------------------------\n');
    
    valid_lyap_idx = ~isnan(emergence_index.lyapunov_estimate);
    if any(valid_lyap_idx)
        valid_lyap = emergence_index.lyapunov_estimate(valid_lyap_idx);
        fprintf(fid, 'Lyapunov Exponent Statistics:\n');
        fprintf(fid, '  Mean: %.4f (std: %.4f)\n', emergence_index.mean_lyapunov, emergence_index.std_lyapunov);
        fprintf(fid, '  Range: [%.4f, %.4f]\n', min(valid_lyap), max(valid_lyap));
        fprintf(fid, '\n');
    else
        fprintf(fid, 'Lyapunov Exponent: N/A (power-law estimation used)\n\n');
    end
    
    for i = 1:num_samples
        sample_name = sample_names{i};
        if ~isempty(all_lyapunov.(sample_name))
            fprintf(fid, 'Sample %s (direct estimation):\n', sample_name);
            fprintf(fid, '  Lambda: %.4f\n', all_lyapunov.(sample_name).lambda);
            fprintf(fid, '  Confidence: %.2f\n', all_lyapunov.(sample_name).confidence);
            if isfield(all_lyapunov.(sample_name), 'method')
                fprintf(fid, '  Method: %s\n', all_lyapunov.(sample_name).method);
            end
            if isfield(all_lyapunov.(sample_name), 'avg_sensitivity')
                fprintf(fid, '  Avg Sensitivity: %.4f\n', all_lyapunov.(sample_name).avg_sensitivity);
            end
        elseif ~isempty(all_powerlaw.(sample_name))
            fprintf(fid, 'Sample %s (power-law inferred):\n', sample_name);
            fprintf(fid, '  Mean Alpha: %.4f\n', all_powerlaw.(sample_name).mean_alpha);
            fprintf(fid, '  Inferred Lyapunov: %.4f\n', all_powerlaw.(sample_name).inferred_lyapunov);
        end
        fprintf(fid, '\n');
    end
    
    fprintf(fid, 'SECTION 4: EMERGENCE INDEX RANKING\n');
    fprintf(fid, '------------------------------------------------------------\n');
    fprintf(fid, 'Combined Emergence Index = Shannon Entropy * (1 + |Lyapunov|)\n\n');
    fprintf(fid, 'Rank  Sample              Combined    Shannon    Lyapunov\n');
    fprintf(fid, '      Index               Index       Entropy    Estimate\n');
    fprintf(fid, '------------------------------------------------------------\n');
    
    [~, sorted_idx] = sort(emergence_index.combined_index, 'descend');
    for rank = 1:num_samples
        idx = sorted_idx(rank);
        fprintf(fid, '%4d  %-20s %10.4f  %10.4f  %10.4f\n', ...
            rank, sample_names{idx}, ...
            emergence_index.combined_index(idx), ...
            emergence_index.shannon_entropy(idx), ...
            emergence_index.lyapunov_estimate(idx));
    end
    fprintf(fid, '\n');
    
    fprintf(fid, '============================================================\n');
    fprintf(fid, 'END OF REPORT\n');
    fprintf(fid, '============================================================\n');
    
    fclose(fid);
end
