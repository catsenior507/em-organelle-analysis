
% 从特征提取阶段开始运行
% 包含所有需要的函数
clear all;
close all;
clc;

diary('feature_extraction.log');
diary on;

fprintf('====================================\n');
fprintf('从特征提取阶段开始\n');
fprintf('====================================\n');

base_path = pwd;
data_path = fullfile(base_path, '..', 'data');
change_img_path = fullfile(data_path, 'change');
change_feature_json_path = fullfile(data_path, 'change_feature_json');
change_features_all_path = fullfile(data_path, 'change_features_all');
change_features_pre_path = fullfile(data_path, 'change_features_pre');
change_mode_path = fullfile(base_path, '..', 'change_mode');

fprintf('检查文件夹...\n');
if ~exist(change_features_all_path, 'dir')
    mkdir(change_features_all_path);
end
if ~exist(change_features_pre_path, 'dir')
    mkdir(change_features_pre_path);
end
if ~exist(change_mode_path, 'dir')
    mkdir(change_mode_path);
end

% 检查是否已经有特征文件
feature_files = dir(fullfile(change_features_all_path, '*.json'));
if length(feature_files) < 1000
    fprintf('\n开始提取增强图片的特征...\n');
    extract_features_from_augmented_local(change_img_path, change_feature_json_path, change_features_all_path);
else
    fprintf('\n已找到 %d 个特征文件，跳过特征提取阶段\n', length(feature_files));
end

fprintf('\n开始训练模型...\n');
train_model_local(change_features_all_path, change_features_pre_path, change_mode_path, data_path);

fprintf('\n====================================\n');
fprintf('完成！\n');
fprintf('====================================\n');

diary off;


function extract_features_from_augmented_local(img_folder, json_folder, output_folder)
    json_files = dir(fullfile(json_folder, '*.json'));
    
    if isempty(json_files)
        fprintf('未找到增强JSON文件\n');
        return;
    end
    
    fprintf('找到 %d 个增强JSON文件\n', length(json_files));
    
    for i = 1:length(json_files)
        json_file = fullfile(json_folder, json_files(i).name);
        [~, json_name, ~] = fileparts(json_files(i).name);
        
        if mod(i, 50) == 0 || i == 1
            fprintf('处理增强文件 %d/%d: %s\n', i, length(json_files), json_files(i).name);
        end
        
        if startsWith(json_name, 'region_analysis__')
            img_name = json_name(length('region_analysis__') + 1:end);
        else
            img_name = json_name;
        end
        
        img_file_tif = fullfile(img_folder, [img_name, '.tif']);
        img_file_tiff = fullfile(img_folder, [img_name, '.tiff']);
        
        if exist(img_file_tif, 'file')
            img_file = img_file_tif;
        elseif exist(img_file_tiff, 'file')
            img_file = img_file_tiff;
        else
            continue;
        end
        
        try
            img = imread(img_file);
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
        catch
            continue;
        end
        
        try
            fid = fopen(json_file, 'r');
            content = fread(fid, '*char')';
            fclose(fid);
        catch
            continue;
        end
        
        features = [];
        try
            json_data = jsondecode(content);
            if ~isfield(json_data, 'regions') || isempty(json_data.regions)
                continue;
            end
            regions_data = json_data.regions;
        catch
            continue;
        end
        
        for j = 1:length(regions_data)
            reg = regions_data(j);
            
            if ~isfield(reg, 'region_pixels') || isempty(reg.region_pixels)
                continue;
            end
            
            coords = reg.region_pixels;
            if size(coords, 2) ~= 2
                continue;
            end
            
            x = coords(:, 1);
            y = coords(:, 2);
            
            min_x = min(x);
            max_x = max(x);
            min_y = min(y);
            max_y = max(y);
            width = max_x - min_x + 1;
            height = max_y - min_y + 1;
            
            center_x = mean(x);
            center_y = mean(y);
            
            major_axis = 0;
            minor_axis = 0;
            if length(x) >= 2
                coords_centered = [x - center_x, y - center_y];
                cov_matrix = cov(coords_centered);
                [~, eigenvalues] = eig(cov_matrix);
                eigenvalues = diag(eigenvalues);
                eigenvalues = sort(eigenvalues, 'descend');
                major_axis = 2 * sqrt(eigenvalues(1));
                minor_axis = 2 * sqrt(eigenvalues(2));
            end
            
            area = length(x);
            roughness = 0;
            if area > 0
                binary_img = false(max_y, max_x);
                for k = 1:length(x)
                    if y(k) <= size(binary_img, 1) && x(k) <= size(binary_img, 2)
                        binary_img(y(k), x(k)) = true;
                    end
                end
                
                if any(any(binary_img))
                    perimeter = sum(bwperim(binary_img, 8));
                    if perimeter > 0
                        roughness = perimeter / sqrt(area);
                    end
                end
            end
            
            gray_values = [];
            for k = 1:length(x)
                if y(k) <= size(img, 1) && x(k) <= size(img, 2)
                    gray_values = [gray_values, double(img(y(k), x(k)))];
                end
            end
            
            mean_gray = 0;
            std_gray = 0;
            if ~isempty(gray_values) && isnumeric(gray_values)
                mean_gray = mean(gray_values);
                if length(gray_values) > 1
                    std_gray = std(gray_values);
                end
            end
            
            complexity = 0;
            if length(x) >= 10
                binary_img = false(max_y, max_x);
                for k = 1:length(x)
                    if y(k) <= size(binary_img, 1) && x(k) <= size(binary_img, 2)
                        binary_img(y(k), x(k)) = true;
                    end
                end
                
                if any(any(binary_img))
                    signal = sum(binary_img, 2);
                    signal = signal(:);
                    
                    fft_result = fft(signal);
                    power_spectrum = abs(fft_result).^2;
                    
                    if length(power_spectrum) > 1
                        [~, max_idx] = max(power_spectrum(2:end));
                        main_frequency = max_idx / length(signal);
                        complexity = main_frequency;
                    end
                end
            end
            
            feature = struct('region_id', reg.region_id, 'recognized_text', '', ...
                'position', struct('min_x', min_x, 'max_x', max_x, 'min_y', min_y, 'max_y', max_y, 'width', width, 'height', height), ...
                'center', struct('x', center_x, 'y', center_y), ...
                'axes', struct('major_axis', major_axis, 'minor_axis', minor_axis), ...
                'roughness', roughness, 'complexity', complexity, ...
                'gray_level', struct('mean', mean_gray, 'std', std_gray));
            features = [features, feature];
        end
        
        if ~isempty(features)
            output_data = struct('regions', features);
            json_output = jsonencode(output_data);
            output_file = fullfile(output_folder, [json_name, '.json']);
            fid = fopen(output_file, 'w');
            fwrite(fid, json_output);
            fclose(fid);
        end
    end
    
    fprintf('特征提取完成！\n');
end


function train_model_local(features_all_path, features_pre_path, mode_path, data_path)
    json_files = dir(fullfile(features_all_path, 'region_analysis__*.json'));
    num_files = length(json_files);
    fprintf('找到 %d 个JSON文件用于训练\n', num_files);
    
    if num_files == 0
        error('没有找到JSON文件用于训练');
    end
    
    all_images = [];
    all_features = [];
    feature_dim = 0;
    
    for i = 1:num_files
        if mod(i, 100) == 0
            fprintf('加载数据进度: %d/%d\n', i, num_files);
        end
        
        json_file = json_files(i).name;
        json_file_path = fullfile(features_all_path, json_file);
        
        [~, base_name, ~] = fileparts(json_file);
        img_name = strrep(base_name, 'region_analysis__', '');
        
        img_file = fullfile(data_path, 'change', [img_name, '.tif']);
        if ~exist(img_file, 'file')
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
        catch
            continue;
        end
        
        try
            json_data = jsondecode(fileread(json_file_path));
            regions = json_data.regions;
        catch
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
    
    indices = randperm(num_samples);
    train_indices = indices(1:train_size);
    test_indices = indices(train_size+1:end);
    
    train_images = all_images(train_indices, :);
    train_features = all_features(train_indices, :);
    test_images = all_images(test_indices, :);
    test_features = all_features(test_indices, :);
    
    fprintf('训练集大小: %d, 测试集大小: %d\n', train_size, test_size);
    
    input_size = size(train_images, 2);
    output_size = size(train_features, 2);
    
    fprintf('输入大小: %d, 输出大小: %d\n', input_size, output_size);
    
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
        'MaxEpochs', 10, ...
        'MiniBatchSize', 8, ...
        'Verbose', true, ...
        'Plots', 'none');
    
    fprintf('====================================\n');
    fprintf('开始训练模型...\n');
    fprintf('====================================\n');
    
    try
        net = trainNetwork(train_images, train_features, layers, options);
        fprintf('模型训练成功\n');
    catch ME
        fprintf('模型训练失败: %s\n', ME.message);
        return;
    end
    
    model_file = fullfile(mode_path, 'representation_learning_model.mat');
    fprintf('保存模型到: %s\n', model_file);
    save(model_file, 'net');
    
    try
        predicted_features = predict(net, test_images);
        fprintf('模型预测成功\n');
    catch ME
        fprintf('模型预测失败: %s\n', ME.message);
        return;
    end
    
    mse = mean((predicted_features - test_features).^2, 'all');
    rmse = sqrt(mse);
    
    fprintf('模型测试结果：\n');
    fprintf('均方误差 (MSE): %.4f\n', mse);
    fprintf('均方根误差 (RMSE): %.4f\n', rmse);
    
    for i = 1:min(test_size, 10)
        pred_result = struct();
        pred_result.regions = {};
        
        region = struct();
        region.region_id = i;
        region.recognized_text = '';
        
        region.position = struct();
        region.position.min_x = predicted_features(i, 1);
        region.position.max_x = predicted_features(i, 2);
        region.position.min_y = predicted_features(i, 3);
        region.position.max_y = predicted_features(i, 4);
        region.position.width = predicted_features(i, 5);
        region.position.height = predicted_features(i, 6);
        
        region.center = struct();
        region.center.x = predicted_features(i, 7);
        region.center.y = predicted_features(i, 8);
        
        region.axes = struct();
        region.axes.major_axis = predicted_features(i, 9);
        region.axes.minor_axis = predicted_features(i, 10);
        
        region.roughness = predicted_features(i, 11);
        region.complexity = predicted_features(i, 12);
        
        region.gray_level = struct();
        region.gray_level.mean = predicted_features(i, 13);
        region.gray_level.std = predicted_features(i, 14);
        
        pred_result.regions{end+1} = region;
        
        pred_file = fullfile(features_pre_path, sprintf('prediction_%d.json', i));
        json_str = jsonencode(pred_result);
        fid = fopen(pred_file, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
    end
    
    fprintf('预测结果保存完成\n');
end
