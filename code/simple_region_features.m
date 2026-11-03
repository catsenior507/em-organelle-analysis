% 保留用户显式设置的开关（原来这里是 clear;，会把开关一起清掉）
clearvars -except ALLOW_LEGACY_BROKEN_INPUT;
clc;

% 把 code 目录加入搜索路径，并按本文件位置解析项目路径
addpath(fileparts(mfilename('fullpath')));
P = project_paths();

% ==========================================================================
% 【停用保护】本脚本读取的 data\features_json 是**已知损坏**的数据：
%   * 91/94 个 JSON 文件里被写入了字面量 `...`，不是合法 JSON；
%   * 每个区域的 region_pixels 被截断到前 1000 个坐标（见 analyze_annotated_regions.m）；
%   * 上游分割用的是 ~green_mask（背景）而不是轮廓内部，坐标语义还是 (row, col)。
% 正确的特征必须使用 tools/morpho/extract_regions.py 生成的 data\annotations_v2：
%     python tools/morpho/extract_regions.py --separated "data/分离的标注信息" ^
%         --originals "data/原图片" --scale-dir "data/scale_mat" --out "data/annotations_v2"
% 如确需复现历史结果，请先执行  ALLOW_LEGACY_BROKEN_INPUT = true;  再运行本脚本。
% ==========================================================================
if ~exist('ALLOW_LEGACY_BROKEN_INPUT', 'var') || ~isequal(ALLOW_LEGACY_BROKEN_INPUT, true)
    warning('simple_region_features:deprecatedInput', ...
        ['本脚本读取的 data\\features_json 是已知损坏的数据（多数文件不是合法 JSON，' ...
         '且区域像素被截断到 1000 点），已停止运行。\n' ...
         '请改用 tools/morpho/extract_regions.py 生成的 data\\annotations_v2：\n  %s\n' ...
         '如确需复现历史结果：先执行 ALLOW_LEGACY_BROKEN_INPUT = true; 再运行本脚本。'], ...
        P.annotations_v2);
    return;
end

% 设置文件夹路径（绝对路径，不再依赖 pwd）
json_folder = P.features_json;
img_folder = P.original_images;
output_folder = P.features_all;

% 确保输出文件夹存在
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 获取所有JSON文件
json_files = dir(fullfile(json_folder, '*.json'));

if isempty(json_files)
    disp('未找到JSON文件');
    return;
end

disp(['找到 ', num2str(length(json_files)), ' 个JSON文件']);

% 遍历每个JSON文件
for i = 1:length(json_files)
    json_file = fullfile(json_folder, json_files(i).name);
    [~, json_name, ~] = fileparts(json_files(i).name);
    
    disp(['处理文件 ', num2str(i), '/', num2str(length(json_files)), ': ', json_files(i).name]);
    
    % 从JSON文件名提取图片文件名（去掉region_analysis__前缀）
    if startsWith(json_name, 'region_analysis__')
        img_name = json_name(length('region_analysis__') + 1:end);
    else
        img_name = json_name;
    end
    
    % 构建图片文件路径（支持.tif和.tiff格式）
    img_file_tif = fullfile(img_folder, [img_name, '.tif']);
    img_file_tiff = fullfile(img_folder, [img_name, '.tiff']);
    
    % 检查图片文件是否存在
    if exist(img_file_tif, 'file')
        img_file = img_file_tif;
    elseif exist(img_file_tiff, 'file')
        img_file = img_file_tiff;
    else
        err_msg = ['警告: 未找到对应图片文件: ', img_name, '.tif/.tiff，跳过该JSON文件'];
        disp(err_msg);
        % 原来是 errordlg：弹出模态对话框，批量/无人值守运行会卡死，改为 warning
        warning('simple_region_features:missingImage', '%s', err_msg);
        continue;
    end
    
    % 读取原图片
    try
        img = imread(img_file);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
    catch ME
        err_msg = ['警告: 读取图片失败: ', img_file, ' - ', ME.message, '，跳过该JSON文件'];
        disp(err_msg);
        % 原为 errordlg（模态对话框，破坏无界面运行）
        warning('simple_region_features:imageReadFailed', '%s', err_msg);
        continue;
    end
    
    % 读取 JSON 文件
    try
        fid = fopen(json_file, 'r');
        content = fread(fid, '*char')';
        fclose(fid);
    catch ME
        err_msg = ['警告: 读取JSON文件失败: ', json_file, ' - ', ME.message, '，跳过该文件'];
        disp(err_msg);
        % 原为 errordlg（模态对话框，破坏无界面运行）
        warning('simple_region_features:jsonReadFailed', '%s', err_msg);
        continue;
    end
    
    % 存储结果
    features = [];
    
    % 查找所有区域
    % 使用更精确的正则表达式，同时提取recognized_text
    region_pattern = '"region_id":\s*(\d+)[\s\S]*?"region_pixels":\s*\[([\s\S]*?)\],\s*"yellow_pixels_coords"[\s\S]*?"recognized_text":\s*"([^"]*)"';
    matches = regexp(content, region_pattern, 'tokens');
    
    % 如果找不到，尝试另一种格式（没有yellow_pixels_coords）
    if isempty(matches)
        region_pattern = '"region_id":\s*(\d+)[\s\S]*?"region_pixels":\s*\[([\s\S]*?)\][\s\S]*?"recognized_text":\s*"([^"]*)"';
        matches = regexp(content, region_pattern, 'tokens');
    end
    
    % 如果还是找不到，尝试只提取region_id和region_pixels
    if isempty(matches)
        region_pattern = '"region_id":\s*(\d+)[\s\S]*?"region_pixels":\s*\[([\s\S]*?)\],\s*"yellow_pixels_coords"';
        matches = regexp(content, region_pattern, 'tokens');
        % 为每个匹配添加空的recognized_text
        for m = 1:length(matches)
            matches{m}{3} = '';
        end
    end
    
    % 最后尝试最简单的格式
    if isempty(matches)
        region_pattern = '"region_id":\s*(\d+)[\s\S]*?"region_pixels":\s*\[([\s\S]*?)\]';
        matches = regexp(content, region_pattern, 'tokens');
        % 为每个匹配添加空的recognized_text
        for m = 1:length(matches)
            matches{m}{3} = '';
        end
    end
    
    if isempty(matches)
        disp(['文件 ', json_files(i).name, ' 未找到区域信息，跳过']);
        continue;
    end
    
    disp(['  找到 ', num2str(length(matches)), ' 个区域']);
    
    % 处理每个区域
    for j = 1:length(matches)
        match = matches{j};
        region_id = str2double(match{1});
        pixels_str = match{2};
        recognized_text = match{3};
        
        % 提取像素坐标
        coords = [];
        % 查找所有坐标对 [x,y]
        coord_pattern = '\[(\d+),(\d+)\]';
        coord_matches = regexp(pixels_str, coord_pattern, 'tokens');
        
        for k = 1:length(coord_matches)
            coord_match = coord_matches{k};
            x = str2double(coord_match{1});
            y = str2double(coord_match{2});
            coords = [coords; x, y];
        end
        
        if isempty(coords)
            disp(['  区域 ', num2str(region_id), ' 没有像素数据，跳过']);
            continue;
        end
        
        % 计算特征
        x = coords(:, 1);
        y = coords(:, 2);
        
        % 位置
        min_x = min(x);
        max_x = max(x);
        min_y = min(y);
        max_y = max(y);
        width = max_x - min_x + 1;
        height = max_y - min_y + 1;
        
        % 中心点
        center_x = mean(x);
        center_y = mean(y);
        
        % 主轴和次轴
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
        
        % 粗糙程度
        area = length(x);
        roughness = 0;
        if area > 0
            % 创建二值图像
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
        
        % 灰度统计
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
        
        % 计算区域复杂程度（使用傅里叶分析的主频率）
        complexity = 0;
        if length(x) >= 10
            % 创建区域的二值图像
            binary_img = false(max_y, max_x);
            for k = 1:length(x)
                if y(k) <= size(binary_img, 1) && x(k) <= size(binary_img, 2)
                    binary_img(y(k), x(k)) = true;
                end
            end
            
            % 计算傅里叶变换
            if any(any(binary_img))
                % 将二值图像转换为一维信号（沿行方向）
                signal = sum(binary_img, 2);
                signal = signal(:);
                
                % 傅里叶变换
                fft_result = fft(signal);
                
                % 计算功率谱
                power_spectrum = abs(fft_result).^2;
                
                % 找到主频率（忽略直流分量）
                if length(power_spectrum) > 1
                    [~, max_idx] = max(power_spectrum(2:end));
                    main_frequency = max_idx / length(signal);
                    complexity = main_frequency;
                end
            end
        end
        
        % 存储特征
        feature = struct('region_id', region_id, 'recognized_text', recognized_text, 'position', struct('min_x', min_x, 'max_x', max_x, 'min_y', min_y, 'max_y', max_y, 'width', width, 'height', height), 'center', struct('x', center_x, 'y', center_y), 'axes', struct('major_axis', major_axis, 'minor_axis', minor_axis), 'roughness', roughness, 'complexity', complexity, 'gray_level', struct('mean', mean_gray, 'std', std_gray));
        features = [features, feature];
        
        disp(['  处理区域 ', num2str(j), '/', num2str(length(matches))]);
    end
    
    % 保存结果
    if ~isempty(features)
        output_data = struct('regions', features);
        json_output = jsonencode(output_data, 'PrettyPrint', true);
        output_file = fullfile(output_folder, [json_name, '.json']);
        fid = fopen(output_file, 'w');
        fwrite(fid, json_output);
        fclose(fid);
        disp(['  特征提取完成，结果已保存到 ', output_file]);
        disp(['  共处理 ', num2str(length(features)), ' 个区域']);
    else
        disp(['  文件 ', json_files(i).name, ' 没有处理任何区域']);
    end
end

disp('所有文件处理完成！');