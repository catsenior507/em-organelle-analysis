% 把 code 目录加入搜索路径，并按本文件位置解析项目路径
addpath(fileparts(mfilename('fullpath')));
P = project_paths();

% ==========================================================================
% 【停用保护】本脚本解析的是旧流水线的 data\分离的标注信息\region_analysis.json，
% 该数据不可用：
%   * 区域像素被截断到前 1000 个坐标，且 JSON 里被写入字面量 `...`（非法 JSON）；
%   * 上游分割用的是 ~green_mask（背景）而不是轮廓内部，坐标语义是 (row, col)，
%     与下游"x=列, y=行"的读法相反；
%   * 下面三行路径 'data/...' 也是错的：历史用法是以 code\ 为工作目录，
%     它们实际指向 code\data\...，目录并不存在。
% 正确做法：使用 tools/morpho/extract_regions.py 生成 data\annotations_v2
% （1-based (x, y) 坐标、合法 JSON、轮廓为简化多边形而非截断像素表）：
%     python tools/morpho/extract_regions.py --separated "data/分离的标注信息" ^
%         --originals "data/原图片" --scale-dir "data/scale_mat" --out "data/annotations_v2"
% 如确需复现历史结果，请先执行  ALLOW_LEGACY_BROKEN_INPUT = true;  再运行本脚本。
% ==========================================================================
if ~exist('ALLOW_LEGACY_BROKEN_INPUT', 'var') || ~isequal(ALLOW_LEGACY_BROKEN_INPUT, true)
    warning('extract_region_features:deprecatedInput', '%s', ...
        ['本脚本解析的旧区域 JSON（region_analysis.json）是已知损坏的数据：' ...
         '区域像素被截断到 1000 点、JSON 内含字面量 ... 因而非法，已停止运行。' newline ...
         '请改用 tools/morpho/extract_regions.py 生成的 data\annotations_v2' newline ...
         '（本机路径：' P.annotations_v2 '）。' newline ...
         '如确需复现历史结果：先执行 ALLOW_LEGACY_BROKEN_INPUT = true; 再运行本脚本。']);
    return;
end

% 读取文件路径（以下为历史遗留路径，仅在显式开启 ALLOW_LEGACY_BROKEN_INPUT 时使用）
json_file = 'data/分离的标注信息/region_analysis.json';
img_file = 'data/原图片/11_0002.tif';
output_file = 'data/region_features.json';

% 读取图片
img = imread(img_file);
if size(img, 3) > 1
    img = rgb2gray(img);
end

% 读取JSON文件内容
fid = fopen(json_file, 'r');
content = fscanf(fid, '%c', inf);
fclose(fid);

% 初始化结果
result = struct('regions', {});

% 提取区域信息
% 首先提取所有区域块
region_blocks = regexp(content, '"region_id":\s*\d+[\s\S]*?"recognized_text":\s*"[^"]*"', 'match');

fprintf('找到 %d 个区域\n', length(region_blocks));

for i = 1:length(region_blocks)
    block = region_blocks{i};
    
    % 提取region_id
    id_match = regexp(block, '"region_id":\s*(\d+)', 'tokens');
    if isempty(id_match)
        continue;
    end
    region_id = str2double(id_match{1}{1});
    
    % 提取region_pixels
    pixels_match = regexp(block, '"region_pixels":\s*\[([\s\S]*?)\]', 'tokens');
    if isempty(pixels_match)
        continue;
    end
    pixels_str = pixels_match{1}{1};
    
    % 提取像素坐标
    pixel_matches = regexp(pixels_str, '\[(\d+),(\d+)\]', 'tokens');
    num_pixels = length(pixel_matches);
    
    if num_pixels == 0
        continue;
    end
    
    fprintf('处理区域 %d，包含 %d 个像素\n', region_id, num_pixels);
    
    % 存储像素坐标
    x_coords = zeros(num_pixels, 1);
    y_coords = zeros(num_pixels, 1);
    
    for j = 1:num_pixels
        x_coords(j) = str2double(pixel_matches{j}{1});
        y_coords(j) = str2double(pixel_matches{j}{2});
    end
    
    % 计算区域位置（边界框）
    min_x = min(x_coords);
    max_x = max(x_coords);
    min_y = min(y_coords);
    max_y = max(y_coords);
    position = [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1];
    
    % 计算中心点
    center_x = mean(x_coords);
    center_y = mean(y_coords);
    center = [center_x, center_y];
    
    % 计算主轴和次轴长度
    major_axis = 0;
    minor_axis = 0;
    if num_pixels > 1
        coords = [x_coords, y_coords];
        cov_mat = cov(coords);
        [V, D] = eig(cov_mat);
        eigenvalues = diag(D);
        [sorted_eig, idx] = sort(eigenvalues, 'descend');
        major_axis = 2 * sqrt(sorted_eig(1));
        minor_axis = 2 * sqrt(sorted_eig(2));
    end
    
    % 计算粗糙程度（使用轮廓的周长与面积的比值）
    binary_img = false(size(img));
    for j = 1:num_pixels
        x = x_coords(j);
        y = y_coords(j);
        if x >= 1 && x <= size(img, 2) && y >= 1 && y <= size(img, 1)
            binary_img(y, x) = true;
        end
    end
    
    area = sum(binary_img(:));
    roughness = 0;
    if area > 0
        stats = regionprops(binary_img, 'Perimeter');
        if ~isempty(stats)
            perimeter = stats.Perimeter;
            roughness = perimeter / sqrt(area);
        end
    end
    
    % 计算内部灰度
    gray_values = [];
    for j = 1:num_pixels
        x = x_coords(j);
        y = y_coords(j);
        if x >= 1 && x <= size(img, 2) && y >= 1 && y <= size(img, 1)
            gray_values = [gray_values; img(y, x)];
        end
    end
    
    mean_gray = 0;
    std_gray = 0;
    if ~isempty(gray_values)
        mean_gray = mean(gray_values);
        std_gray = std(gray_values);
    end
    
    % 构建区域特征结构
    region_feature = struct('region_id', region_id, 'position', position, 'center', center, 'major_axis', major_axis, 'minor_axis', minor_axis, 'roughness', roughness, 'mean_gray', mean_gray, 'std_gray', std_gray);
    
    % 添加到结果中
    result.regions(i) = region_feature;
    fprintf('区域 %d 处理完成\n', region_id);
end

% 保存结果
json_result = jsonencode(result, 'PrettyPrint', true);
fid = fopen(output_file, 'w');
fwrite(fid, json_result);
fclose(fid);

fprintf('特征提取完成，结果已保存到 %s\n', output_file);