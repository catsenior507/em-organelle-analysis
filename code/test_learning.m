
% 测试脚本 - 只处理前10个文件
clear all;
close all;
clc;

diary('test_learning.log');
diary on;

fprintf('====================================\n');
fprintf('测试脚本 - 只处理前10个文件\n');
fprintf('====================================\n');

base_path = pwd;
data_path = fullfile(base_path, '..', 'data');
change_img_path = fullfile(data_path, 'change');
change_feature_json_path = fullfile(data_path, 'change_feature_json');
change_features_all_path = fullfile(data_path, 'change_features_all');
change_features_pre_path = fullfile(data_path, 'change_features_pre_test');
change_mode_path = fullfile(base_path, '..', 'change_mode_test');

if ~exist(change_features_pre_path, 'dir')
    mkdir(change_features_pre_path);
end
if ~exist(change_mode_path, 'dir')
    mkdir(change_mode_path);
end

fprintf('\n开始训练模型 - 只使用前10个文件\n');
json_files = dir(fullfile(change_features_all_path, 'region_analysis__*.json'));
num_files = min(10, length(json_files));
fprintf('找到 %d 个JSON文件，使用前 %d 个\n', length(json_files), num_files);

if num_files == 0
    error('没有找到JSON文件用于训练');
end

all_images = [];
all_features = [];
feature_dim = 0;

for i = 1:num_files
    fprintf('加载数据进度: %d/%d\n', i, num_files);

    json_file = json_files(i).name;
    json_file_path = fullfile(change_features_all_path, json_file);

    [~, base_name, ~] = fileparts(json_file);
    image_name = strrep(base_name, 'region_analysis__', '');

    img_file = fullfile(data_path, 'change', [image_name, '.tif']);
    if ~exist(img_file, 'file')
        fprintf('警告：找不到图片 %s\n', img_file);
        continue;
    end

    try
        img = imread(img_file);
        img_resized = imresize(img, [256, 256]);
        if size(img_resized, 3) == 3
            img_gray = rgb2gray(img_resized);
        else
            img_gray = img_resized;
        end
        img_normalized = double(img_gray) / 255;
    catch ME
        fprintf('警告：图片读取失败 %s - %s\n', img_file, ME.message);
        continue;
    end

    try
        json_data = jsondecode(fileread(json_file_path));
        regions = json_data.regions;
    catch ME
        fprintf('警告：无法解析JSON文件 %s - %s\n', json_file_path, ME.message);
        continue;
    end

    for j = 1:length(regions)
        region = regions(j);

        feature = [];

        % 检查必需字段
        if ~isfield(region, 'position') || ~isfield(region, 'center') || ~isfield(region, 'axes') || ~isfield(region, 'gray_level')
            continue;
        end

        position = region.position;
        feature = [feature, position.min_x, position.max_x, position.min_y, position.max_y, position.width, position.height];

        center = region.center;
        feature = [feature, center.x, center.y];

        axes = region.axes;
        minor_axis_val = 0;
        if isfield(axes, 'minor_axis')
            minor_axis_val = axes.minor_axis;
        elseif isfield(axes, 'min_axis')
            minor_axis_val = axes.min_axis;
        end
        feature = [feature, axes.major_axis, minor_axis_val];

        roughness = 0;
        complexity = 0;
        if isfield(region, 'roughness') && ~isempty(region.roughness)
            val = region.roughness;
            if isscalar(val)
                roughness = val;
            elseif isnumeric(val) && length(val) == 1
                roughness = double(val);
            end
        end
        if isfield(region, 'complexity') && ~isempty(region.complexity)
            val = region.complexity;
            if isscalar(val)
                complexity = val;
            elseif isnumeric(val) && length(val) == 1
                complexity = double(val);
            end
        end
        feature = [feature, roughness, complexity];

        gray_level = region.gray_level;
        feature = [feature, gray_level.mean, gray_level.std];

        if feature_dim == 0
            feature_dim = length(feature);
        end

        if length(feature) ~= feature_dim
            continue;
        end

        all_images = [all_images; img_normalized(:)'];
        if isempty(all_features)
            all_features = feature;
        else
            all_features = [all_features; feature];
        end
    end
end

fprintf('数据加载完成，共加载 %d 个样本\n', size(all_images, 1));

if size(all_images, 1) == 0
    error('没有加载到有效的样本');
end

num_samples = size(all_images, 1);
train_ratio = 0.8;
train_size = floor(num_samples * train_ratio);
test_size = num_samples - train_size;

if test_size == 0
    test_size = 1;
    train_size = num_samples - 1;
end

fprintf('训练集大小: %d, 测试集大小: %d\n', train_size, test_size);

indices = randperm(num_samples);
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

train_images = all_images(train_indices, :);
train_features = all_features(train_indices, :);
test_images = all_images(test_indices, :);
test_features = all_features(test_indices, :);

input_size = size(train_images, 2);
output_size = size(train_features, 2);

fprintf('输入大小: %d, 输出大小: %d\n', input_size, output_size);

try
    layers = [
        featureInputLayer(input_size)
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(output_size)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 3, ...
        'MiniBatchSize', 8, ...
        'Verbose', true, ...
        'Plots', 'none');

    fprintf('====================================\n');
    fprintf('开始训练模型...\n');
    fprintf('训练参数: MaxEpochs=%d, MiniBatchSize=%d\n', 3, 8);
    fprintf('====================================\n');
    net = trainNetwork(train_images, train_features, layers, options);
    fprintf('模型训练成功\n');

catch ME
    fprintf('模型训练失败: %s\n', ME.message);
    diary off;
    error('模型训练失败');
end

model_file = fullfile(change_mode_path, 'test_learning_model.mat');
fprintf('保存模型到: %s\n', model_file);
save(model_file, 'net');

try
    predicted_features = predict(net, test_images);
    fprintf('模型预测成功\n');
catch ME
    fprintf('模型预测失败: %s\n', ME.message);
    diary off;
    error('模型预测失败');
end

mse = mean((predicted_features - test_features).^2, 'all');
rmse = sqrt(mse);

fprintf('模型测试结果：\n');
fprintf('均方误差 (MSE): %.4f\n', mse);
fprintf('均方根误差 (RMSE): %.4f\n', rmse);

for i = 1:test_size
    sample_features = predicted_features(i, :);
    
    pred_result = struct();
    pred_result.regions = {};
    
    region = struct();
    region.region_id = i;
    region.recognized_text = '';
    
    region.position = struct();
    region.position.min_x = sample_features(1);
    region.position.max_x = sample_features(2);
    region.position.min_y = sample_features(3);
    region.position.max_y = sample_features(4);
    region.position.width = sample_features(5);
    region.position.height = sample_features(6);
    
    region.center = struct();
    region.center.x = sample_features(7);
    region.center.y = sample_features(8);
    
    region.axes = struct();
    region.axes.major_axis = sample_features(9);
    region.axes.minor_axis = sample_features(10);
    
    region.roughness = sample_features(11);
    region.complexity = sample_features(12);
    
    region.gray_level = struct();
    region.gray_level.mean = sample_features(13);
    region.gray_level.std = sample_features(14);
    
    pred_result.regions{end+1} = region;
    
    pred_file = fullfile(change_features_pre_path, sprintf('prediction_%d.json', i));
    json_str = jsonencode(pred_result);
    fid = fopen(pred_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

fprintf('预测结果保存完成\n');

fprintf('\n====================================\n');
fprintf('测试脚本运行成功！\n');
fprintf('====================================\n');

diary off;
