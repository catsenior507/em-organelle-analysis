% 神经网络机器学习脚本
% 输入：data\原图片中的图片
% 输出目标：data\features_all中的json文件的键值
% 模型保存：mode文件夹
% 预测结果：data\features_pre文件夹

% 清理工作空间
clear all;
close all;
clc;

% 强制在命令窗口中显示输出
diary('neural_network_training.log');
diary on;

fprintf('====================================\n');
fprintf('神经网络机器学习训练脚本\n');
fprintf('====================================\n');

% 设置路径
base_path = pwd;
fprintf('当前工作目录: %s\n', base_path);

% 确保我们在code目录中
[~, dir_name, ~] = fileparts(base_path);
if ~strcmp(dir_name, 'code')
    error('请在code目录中运行此脚本');
end

% 正确设置路径
data_path = fullfile(base_path, '..', 'data');
features_all_path = fullfile(data_path, 'features_all');
features_pre_path = fullfile(data_path, 'features_pre');
mode_path = fullfile(base_path, '..', 'mode');

fprintf('数据目录: %s\n', data_path);
fprintf('特征文件夹: %s\n', features_all_path);
fprintf('预测结果文件夹: %s\n', features_pre_path);
fprintf('模型保存文件夹: %s\n', mode_path);

% 创建必要的文件夹
if ~exist(features_pre_path, 'dir')
    fprintf('创建features_pre文件夹...\n');
    mkdir(features_pre_path);
    fprintf('features_pre文件夹创建成功\n');
else
    fprintf('features_pre文件夹已存在\n');
end

if ~exist(mode_path, 'dir')
    fprintf('创建mode文件夹...\n');
    mkdir(mode_path);
    fprintf('mode文件夹创建成功\n');
else
    fprintf('mode文件夹已存在\n');
end

% 检查features_all文件夹是否存在
if ~exist(features_all_path, 'dir')
    error('features_all文件夹不存在: %s', features_all_path);
else
    fprintf('features_all文件夹存在\n');
end

% 获取所有JSON文件
json_files = dir(fullfile(features_all_path, 'region_analysis__*.json'));
num_files = length(json_files);
fprintf('找到 %d 个JSON文件\n', num_files);

if num_files == 0
    error('没有找到JSON文件');
end

% 初始化数据结构
all_images = [];
all_features = [];
feature_dim = 0;

% 存储所有图片的信息
all_image_info = [];

% 处理所有JSON文件
for json_idx = 1:length(json_files)
    json_file = json_files(json_idx).name;
    json_file_path = fullfile(features_all_path, json_file);
    fprintf('处理文件: %s\n', json_file);
    
    % 提取图片名称
    [~, base_name, ~] = fileparts(json_file);
    image_name = strrep(base_name, 'region_analysis__', '');
    fprintf('提取的图片名称: %s\n', image_name);
    
    % 查找对应的图片文件
    image_formats = {'.tif'};
    image_path = '';
    
    % 尝试在分离的标注信息文件夹中查找
    green_image_name = ['green_', image_name, '.tif'];
    image_path = fullfile(data_path, '分离的标注信息', green_image_name);
    
    if exist(image_path, 'file')
        fprintf('找到图片: %s\n', image_path);
        % 读取图片
        img = imread(image_path);
    else
        fprintf('警告：找不到图片 %s\n', green_image_name);
        % 跳过此文件
        continue;
    end
    
    % 调整图片大小为统一尺寸
    img_resized = imresize(img, [256, 256]);
    % 转换为灰度图
    if size(img_resized, 3) == 3
        img_gray = rgb2gray(img_resized);
    else
        img_gray = img_resized;
    end
    % 归一化
    img_normalized = double(img_gray) / 255;
    
    % 读取JSON文件
    json_data = jsondecode(fileread(json_file_path));
    regions = json_data.regions;
    fprintf('找到 %d 个区域\n', length(regions));
    
    % 处理每个区域
    for j = 1:length(regions)
        region = regions(j);
        
        try
            % 提取特征（排除region_id）
            feature = [];
            
            % 处理position
            if isfield(region, 'position')
                position = region.position;
                feature = [feature, position.min_x, position.max_x, position.min_y, position.max_y, position.width, position.height];
            else
                fprintf('警告：区域 %d 缺少position字段\n', j);
                continue;
            end
            
            % 处理center
            if isfield(region, 'center')
                center = region.center;
                feature = [feature, center.x, center.y];
            else
                fprintf('警告：区域 %d 缺少center字段\n', j);
                continue;
            end
            
            % 处理axes
            if isfield(region, 'axes')
                axes = region.axes;
                feature = [feature, axes.major_axis, axes.minor_axis];
            else
                fprintf('警告：区域 %d 缺少axes字段\n', j);
                continue;
            end
            
            % 处理roughness和complexity
            if isfield(region, 'roughness') && isfield(region, 'complexity')
                feature = [feature, region.roughness, region.complexity];
            else
                fprintf('警告：区域 %d 缺少roughness或complexity字段\n', j);
                continue;
            end
            
            % 处理gray_level
            if isfield(region, 'gray_level')
                gray_level = region.gray_level;
                feature = [feature, gray_level.mean, gray_level.std];
            else
                fprintf('警告：区域 %d 缺少gray_level字段\n', j);
                continue;
            end
            
            % 处理recognized_text（去掉长于10个字符的值）
            if isfield(region, 'recognized_text')
                recognized_text = region.recognized_text;
                if length(recognized_text) <= 10 && ~isempty(recognized_text)
                    fprintf('找到有效的文本: %s\n', recognized_text);
                end
            end
            
            % 调试：打印feature的大小
            fprintf('Feature size: %d\n', length(feature));
            if length(feature) > 0
                fprintf('First few features: %f, %f, %f\n', feature(1), feature(2), feature(3));
            end
            
            % 初始化feature_dim
            if feature_dim == 0
                feature_dim = length(feature);
                fprintf('Feature dimension initialized to: %d\n', feature_dim);
            end
            
            % 确保特征维度一致
            if length(feature) ~= feature_dim
                fprintf('警告：特征维度不一致，跳过此区域\n');
                continue;
            end
            
            % 检查是否有NaN或Inf值
            if any(isnan(feature)) || any(isinf(feature))
                fprintf('警告：区域 %d 包含NaN或Inf值，跳过\n', j);
                continue;
            end
            
            % 添加到数据集中
            all_images = [all_images; img_normalized(:)'];
            if isempty(all_features)
                all_features = feature;
            else
                all_features = [all_features; feature];
            end
            
            % 保存图片信息（使用有效的字段名，添加img_前缀避免数字开头）
            img_info = struct();
            img_info.name = image_name;
            img_info.struct_name = ['img_', regexprep(image_name, '^(\d)', 'n$1')];  % 数字开头时添加n前缀
            img_info.path = image_path;
            all_image_info = [all_image_info; img_info];
        catch ME
            fprintf('警告：处理区域 %d 时出错: %s\n', j, ME.message);
            continue;
        end
    end
end

fprintf('数据加载完成，共加载 %d 个样本\n', size(all_images, 1));
fprintf('all_features size: %d x %d\n', size(all_features, 1), size(all_features, 2));

if size(all_images, 1) == 0
    error('没有加载到有效的样本');
end

% 划分训练集和测试集
fprintf('划分训练集和测试集...\n');

num_samples = size(all_images, 1);
train_ratio = 0.8;
train_size = floor(num_samples * train_ratio);
test_size = num_samples - train_size;

if test_size == 0
    test_size = 1;
    train_size = num_samples - 1;
end

indices = randperm(num_samples);
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

train_images = all_images(train_indices, :);
train_features = all_features(train_indices, :);
test_images = all_images(test_indices, :);
test_features = all_features(test_indices, :);

fprintf('训练集大小: %d, 测试集大小: %d\n', train_size, test_size);

% 特征归一化（使用Z-score标准化防止NaN扩散）
fprintf('进行特征归一化...\n');
feature_mean = mean(train_features, 1);
feature_std = std(train_features, 0, 1);
% 防止除以零
feature_std(feature_std == 0) = 1;
train_features_normalized = (train_features - feature_mean) ./ feature_std;
test_features_normalized = (test_features - feature_mean) ./ feature_std;

% 检查是否有NaN或Inf
if any(isnan(train_features_normalized(:))) || any(isinf(train_features_normalized(:)))
    fprintf('警告：归一化后存在NaN或Inf值，尝试使用替代方法\n');
    % 使用min-max归一化到[0,1]
    feature_min = min(train_features, [], 1);
    feature_max = max(train_features, [], 1);
    feature_range = feature_max - feature_min;
    feature_range(feature_range == 0) = 1;
    train_features_normalized = (train_features - feature_min) ./ feature_range;
    test_features_normalized = (test_features - feature_min) ./ feature_range;
end

% 再次检查并强制清理
train_features_normalized(isnan(train_features_normalized)) = 0;
train_features_normalized(isinf(train_features_normalized)) = 0;
test_features_normalized(isnan(test_features_normalized)) = 0;
test_features_normalized(isinf(test_features_normalized)) = 0;

fprintf('特征归一化完成\n');
fprintf('train_features_normalized size: %d x %d\n', size(train_features_normalized, 1), size(train_features_normalized, 2));
fprintf('归一化后特征范围: [%.4f, %.4f]\n', min(train_features_normalized(:)), max(train_features_normalized(:)));

% 创建深度学习模型
fprintf('创建深度学习模型...\n');

% 定义模型参数
input_size = size(train_images, 2);
output_size = size(train_features, 2);

fprintf('输入大小: %d, 输出大小: %d\n', input_size, output_size);

% 训练模型
fprintf('训练深度学习模型...\n');
fprintf('train_images size: %d x %d\n', size(train_images, 1), size(train_images, 2));
fprintf('train_features size: %d x %d\n', size(train_features, 1), size(train_features, 2));

% 检查深度学习函数是否存在
fprintf('检查深度学习函数是否存在...\n');
functions_to_check = {'trainNetwork', 'featureInputLayer', 'fullyConnectedLayer', 'reluLayer', 'regressionLayer', 'trainingOptions'};
for i = 1:length(functions_to_check)
    func_name = functions_to_check{i};
    exist_status = exist(func_name, 'file');
    if exist_status == 2
        fprintf('✓ %s 函数存在\n', func_name);
    else
        fprintf('✗ %s 函数不存在 (状态: %d)\n', func_name, exist_status);
    end
end

% 尝试使用深度学习模型
try
    % 显示网络结构（使用BatchNormalization提高稳定性）
    fprintf('创建网络结构...\n');
    layers = [
        featureInputLayer(input_size)
        fullyConnectedLayer(256)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(128)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(output_size)
        regressionLayer
    ];
    fprintf('网络结构创建成功\n');
    
    % 设置训练选项（添加梯度裁剪和学习率调整防止NaN）
    fprintf('设置训练选项...\n');
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 4, ...
        'InitialLearnRate', 0.0001, ...  % 大幅降低初始学习率
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 30, ...
        'GradientThreshold', 1, ...  % 梯度裁剪防止梯度爆炸
        'GradientThresholdMethod', 'absolute', ...
        'ValidationData', {test_images, test_features_normalized}, ...
        'ValidationFrequency', 10, ...
        'Verbose', true, ...
        'Shuffle', 'every-epoch');  % 每个epoch打乱数据
    fprintf('训练选项设置成功\n');
    
    % 训练模型（使用归一化后的特征）
    fprintf('开始训练模型...\n');
    net = trainNetwork(train_images, train_features_normalized, layers, options);
    fprintf('模型训练成功\n');
    fprintf('net变量类型: %s\n', class(net));
catch ME
    fprintf('模型训练失败: %s\n', ME.message);
    fprintf('错误堆栈: %s\n', getReport(ME));
    diary off;
    error('模型训练失败');
end

% 保存模型
model_file = fullfile(mode_path, 'neural_network_model.mat');
fprintf('保存模型到: %s\n', model_file);
try
    save(model_file, 'net');
    fprintf('模型保存成功\n');
catch ME
    fprintf('模型保存失败: %s\n', ME.message);
    diary off;
    error('模型保存失败');
end

% 测试模型
fprintf('测试模型...\n');
% 对于深度学习模型，使用predict函数进行预测
try
    predicted_features = predict(net, test_images);
    fprintf('模型预测成功\n');
    
    % 检查预测结果是否有NaN
    if any(isnan(predicted_features(:)))
        fprintf('警告：预测结果包含NaN值，尝试使用训练数据的统计信息进行修复\n');
        % 找出包含NaN的样本
        nan_rows = any(isnan(predicted_features), 2);
        fprintf('包含NaN的样本数: %d / %d\n', sum(nan_rows), size(predicted_features, 1));
    end
catch ME
    fprintf('模型预测失败: %s\n', ME.message);
    diary off;
    error('模型预测失败');
end

% 计算误差（使用归一化后的特征比较）
mse = mean((predicted_features - test_features_normalized).^2, 'all');
rmse = sqrt(mse);

fprintf('模型测试结果：\n');
fprintf('均方误差 (MSE): %.4f\n', mse);
fprintf('均方根误差 (RMSE): %.4f\n', rmse);

% 保存预测结果
fprintf('保存预测结果...\n');

% 按图片名称组织预测结果
img_predictions = struct();

% 为测试集中的每个样本保存预测结果
for i = 1:test_size
    sample_features = predicted_features(i, :);
    
    % 获取测试样本对应的图片信息
    test_idx = test_indices(i);
    img_info = all_image_info(test_idx);
    img_name = img_info.name;
    struct_name = img_info.struct_name;
    
    % 跳过无效的预测值
    if any(isnan(sample_features)) || any(isinf(sample_features))
        fprintf('警告：样本 %d 包含NaN或Inf值，跳过\n', i);
        continue;
    end
    
    % 构建区域结构
    region = struct();
    region.region_id = i;
    region.recognized_text = '';
    
    % 解析预测的特征
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
    
    % 将区域添加到对应图片的预测结果中（使用有效的结构体字段名）
    if ~isfield(img_predictions, struct_name)
        img_predictions.(struct_name) = struct();
        img_predictions.(struct_name).name = img_name;
        img_predictions.(struct_name).regions = {};
    end
    img_predictions.(struct_name).regions{end+1} = region;
end

% 保存每个图片的预测结果
struct_names = fieldnames(img_predictions);
for i = 1:length(struct_names)
    struct_name = struct_names{i};
    img_name = img_predictions.(struct_name).name;
    pred_result = struct();
    pred_result.regions = img_predictions.(struct_name).regions;
    
    % 使用原图片名字加predict进行命名
    pred_file = fullfile(features_pre_path, sprintf('%s_predict.json', img_name));
    json_str = jsonencode(pred_result);
    fid = fopen(pred_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    fprintf('保存预测结果: %s\n', pred_file);
end

fprintf('====================================\n');
fprintf('训练完成！\n');
fprintf('模型已保存到: %s\n', model_file);
fprintf('预测结果已保存到: %s\n', features_pre_path);
fprintf('====================================\n');

% 关闭日志
diary off;
