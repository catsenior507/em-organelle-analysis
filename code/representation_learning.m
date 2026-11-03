% 表征化学习脚本
% 功能：对图片进行切割、旋转等增强操作，同步处理JSON区域信息，提取特征并训练模型
% 输入：data\原图片中的图片
% 输出目标：data\change_features_all中的json文件的键值
% 模型保存：change_mode文件夹
% 预测结果：data\change_features_pre文件夹

clear all;
close all;
clc;

diary('representation_learning.log');
diary on;

fprintf('====================================\n');
fprintf('表征化学习脚本\n');
fprintf('====================================\n');

base_path = pwd;
fprintf('当前工作目录: %s\n', base_path);

[~, dir_name, ~] = fileparts(base_path);
if ~strcmp(dir_name, 'code')
    error('请在code目录中运行此脚本');
end

data_path = fullfile(base_path, '..', 'data');
original_img_path = fullfile(data_path, '原图片');
features_json_path = fullfile(data_path, 'features_json');
change_img_path = fullfile(data_path, 'change');
change_feature_json_path = fullfile(data_path, 'change_feature_json');
change_features_all_path = fullfile(data_path, 'change_features_all');
change_features_pre_path = fullfile(data_path, 'change_features_pre');
change_mode_path = fullfile(base_path, '..', 'change_mode');

fprintf('原图片目录: %s\n', original_img_path);
fprintf('特征JSON目录: %s\n', features_json_path);
fprintf('增强图片目录: %s\n', change_img_path);
fprintf('增强特征JSON目录: %s\n', change_feature_json_path);
fprintf('增强特征提取目录: %s\n', change_features_all_path);
fprintf('预测结果目录: %s\n', change_features_pre_path);
fprintf('模型保存目录: %s\n', change_mode_path);

if ~exist(change_img_path, 'dir')
    mkdir(change_img_path);
    fprintf('创建change文件夹成功\n');
end

if ~exist(change_feature_json_path, 'dir')
    mkdir(change_feature_json_path);
    fprintf('创建change_feature_json文件夹成功\n');
end

if ~exist(change_features_all_path, 'dir')
    mkdir(change_features_all_path);
    fprintf('创建change_features_all文件夹成功\n');
end

if ~exist(change_features_pre_path, 'dir')
    mkdir(change_features_pre_path);
    fprintf('创建change_features_pre文件夹成功\n');
end

if ~exist(change_mode_path, 'dir')
    mkdir(change_mode_path);
    fprintf('创建change_mode文件夹成功\n');
end

img_files = dir(fullfile(original_img_path, '*.tif'));
if isempty(img_files)
    img_files = dir(fullfile(original_img_path, '*.tiff'));
end
num_imgs = length(img_files);
fprintf('找到 %d 个原图片文件\n', num_imgs);

if num_imgs == 0
    error('没有找到原图片文件');
end

% 检查是否已经完成了图片增强
augmented_files = dir(fullfile(change_img_path, '*.tif'));
if length(augmented_files) < 1000
    augmentation_count = 0;
else
    fprintf('\n已找到 %d 个增强图片，跳过图片增强阶段\n', length(augmented_files));
    augmentation_count = length(augmented_files);
end

if augmentation_count < 1000

for i = 1:num_imgs
    img_file = fullfile(original_img_path, img_files(i).name);
    [~, img_name, img_ext] = fileparts(img_files(i).name);
    
    fprintf('\n处理图片 %d/%d: %s\n', i, num_imgs, img_files(i).name);
    
    img = imread(img_file);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img);
    
    [img_height, img_width] = size(img);
    fprintf('图片尺寸: %d x %d\n', img_height, img_width);
    
    json_file = fullfile(features_json_path, ['region_analysis__', img_name, '.json']);
    if ~exist(json_file, 'file')
        fprintf('警告: 未找到对应的JSON文件: %s\n', json_file);
        continue;
    end
    
    fid = fopen(json_file, 'r');
    json_content = fread(fid, '*char')';
    fclose(fid);
    
    regions_data = parse_json_regions(json_content);
    
    if isempty(regions_data)
        fprintf('警告: JSON文件没有有效的区域数据: %s\n', json_file);
        fprintf('  JSON内容长度: %d 字符\n', length(json_content));
        continue;
    else
        fprintf('  解析到 %d 个区域\n', length(regions_data));
    end
    
    augmentations = struct();
    
    fprintf('  执行旋转增强...\n');
    rotations = [90, 180, 270];
    for r = 1:length(rotations)
        angle = rotations(r);
        aug_img = imrotate(img, angle, 'bilinear', 'loose');
        [new_height, new_width] = size(aug_img);
        aug_name = sprintf('aug_%s_rot%d', img_name, angle);
        augmentations.(aug_name) = struct('type', 'rotation', 'angle', angle, ...
            'img', aug_img, 'new_width', new_width, 'new_height', new_height);
        augmentation_count = augmentation_count + 1;
    end
    
    fprintf('  执行切割增强...\n');
    crop_sizes = [0.5, 0.75];
    for c = 1:length(crop_sizes)
        ratio = crop_sizes(c);
        new_height = round(img_height * ratio);
        new_width = round(img_width * ratio);
        
        positions = {'center', 'topleft', 'topright', 'bottomleft', 'bottomright'};
        for p = 1:length(positions)
            pos = positions{p};
            switch pos
                case 'center'
                    start_y = round((img_height - new_height) / 2) + 1;
                    start_x = round((img_width - new_width) / 2) + 1;
                case 'topleft'
                    start_y = 1;
                    start_x = 1;
                case 'topright'
                    start_y = 1;
                    start_x = img_width - new_width + 1;
                case 'bottomleft'
                    start_y = img_height - new_height + 1;
                    start_x = 1;
                case 'bottomright'
                    start_y = img_height - new_height + 1;
                    start_x = img_width - new_width + 1;
            end
            
            aug_img = img(start_y:start_y+new_height-1, start_x:start_x+new_width-1);
            aug_name = sprintf('aug_%s_crop%d_%s', img_name, round(ratio*100), pos);
            augmentations.(aug_name) = struct('type', 'crop', 'ratio', ratio, ...
                'start_x', start_x, 'start_y', start_y, ...
                'new_width', new_width, 'new_height', new_height, 'img', aug_img);
            augmentation_count = augmentation_count + 1;
        end
    end
    
    fprintf('  执行翻转增强...\n');
    aug_img_lr = fliplr(img);
    aug_name = sprintf('aug_%s_flip_lr', img_name);
    augmentations.(aug_name) = struct('type', 'flip', 'direction', 'lr', 'img', aug_img_lr);
    augmentation_count = augmentation_count + 1;
    
    aug_img_ud = flipud(img);
    aug_name = sprintf('aug_%s_flip_ud', img_name);
    augmentations.(aug_name) = struct('type', 'flip', 'direction', 'ud', 'img', aug_img_ud);
    augmentation_count = augmentation_count + 1;
    
    aug_names = fieldnames(augmentations);
    fprintf('  生成 %d 个增强图片\n', length(aug_names));
    
    for a = 1:length(aug_names)
        aug_name = aug_names{a};
        aug_info = augmentations.(aug_name);
        aug_img = aug_info.img;
        
        save_name = strrep(aug_name, 'aug_', '');
        
        aug_img_uint8 = uint8(aug_img);
        aug_img_file = fullfile(change_img_path, [save_name, '.tif']);
        imwrite(aug_img_uint8, aug_img_file);
        
        new_json_data = transform_json_regions(regions_data, aug_info, img_height, img_width);
        
        has_regions = isfield(new_json_data, 'regions') && ~isempty(new_json_data.regions);
        if ~has_regions
            fprintf('  增强 %s 转换后区域为空\n', save_name);
        else
            fprintf('  增强 %s 生成 %d 个区域\n', save_name, length(new_json_data.regions));
            new_json_file = fullfile(change_feature_json_path, ['region_analysis__', save_name, '.json']);
            json_str = jsonencode(new_json_data, 'PrettyPrint', true);
            fid = fopen(new_json_file, 'w');
            fprintf(fid, '%s', json_str);
            fclose(fid);
        end
    end
end
end

fprintf('\n====================================\n');
fprintf('图片增强完成，共生成 %d 个增强图片\n', augmentation_count);
fprintf('====================================\n');

% 检查是否已经完成了特征提取
feature_files = dir(fullfile(change_features_all_path, '*.json'));
if length(feature_files) < 1000
    fprintf('\n开始提取增强图片的特征...\n');
    extract_features_from_augmented(change_img_path, change_feature_json_path, change_features_all_path);
else
    fprintf('\n已找到 %d 个特征文件，跳过特征提取阶段\n', length(feature_files));
end

fprintf('\n开始训练模型...\n');
train_model(change_features_all_path, change_features_pre_path, change_mode_path, data_path);

fprintf('\n====================================\n');
fprintf('表征化学习完成！\n');
fprintf('增强图片保存在: %s\n', change_img_path);
fprintf('增强特征JSON保存在: %s\n', change_feature_json_path);
fprintf('特征提取结果保存在: %s\n', change_features_all_path);
fprintf('模型保存在: %s\n', change_mode_path);
fprintf('预测结果保存在: %s\n', change_features_pre_path);
fprintf('====================================\n');

diary off;

function regions_data = parse_json_regions(json_content)
    regions_data = [];
    
    try
        json_content_clean = preclean_json(json_content);
        json_struct = jsondecode(json_content_clean);
        if ~isfield(json_struct, 'regions') || isempty(json_struct.regions)
            fprintf('    JSON中没有regions字段或为空\n');
            return;
        end
        
        num_regions = length(json_struct.regions);
        fprintf('    成功解析JSON，找到 %d 个区域\n', num_regions);
        
        for i = 1:num_regions
            reg = json_struct.regions(i);
            
            region = struct();
            if isfield(reg, 'region_id')
                region.region_id = reg.region_id;
            else
                region.region_id = i;
            end
            
            if isfield(reg, 'region_pixels') && ~isempty(reg.region_pixels)
                pixels = reg.region_pixels;
                if isnumeric(pixels) && size(pixels, 2) == 2
                    region.region_pixels = pixels;
                elseif iscell(pixels)
                    region.region_pixels = cell2mat(pixels);
                else
                    region.region_pixels = [];
                end
            else
                region.region_pixels = [];
            end
            
            if isfield(reg, 'yellow_pixels')
                region.yellow_pixels = reg.yellow_pixels;
            else
                region.yellow_pixels = 0;
            end
            
            if isfield(reg, 'yellow_pixels_coords') && ~isempty(reg.yellow_pixels_coords)
                yellow = reg.yellow_pixels_coords;
                if isnumeric(yellow) && size(yellow, 2) == 2
                    region.yellow_pixels_coords = yellow;
                elseif iscell(yellow)
                    region.yellow_pixels_coords = cell2mat(yellow);
                else
                    region.yellow_pixels_coords = [];
                end
            else
                region.yellow_pixels_coords = [];
            end
            
            if isfield(reg, 'recognized_text')
                if ischar(reg.recognized_text)
                    region.recognized_text = reg.recognized_text;
                else
                    region.recognized_text = '';
                end
            else
                region.recognized_text = '';
            end
            
            regions_data = [regions_data, region];
        end
        
    catch ME
        fprintf('    jsondecode解析失败: %s\n', ME.message);
        fprintf('    尝试备用正则表达式解析方法...\n');
        regions_data = parse_json_regex(json_content);
    end
end

function json_content_clean = preclean_json(json_content)
    json_content_clean = json_content;
    
    json_content_clean = regexprep(json_content_clean, '\r\n', '\n');
    json_content_clean = regexprep(json_content_clean, '\r', '\n');
    
    json_content_clean = remove_ellipsis_truncation(json_content_clean);
    
    json_content_clean = remove_unsupported_escapes(json_content_clean);
    json_content_clean = fix_multiline_text(json_content_clean);
    json_content_clean = fix_json_quotation(json_content_clean);
    json_content_clean = strtrim(json_content_clean);
end

function cleaned = remove_ellipsis_truncation(str)
    cleaned = str;
    
    cleaned = regexprep(cleaned, '\.\.\.(\s*[\],}])', '$1');
    cleaned = regexprep(cleaned, '(\[\s*)\.\.\.', '$1');
    
    cleaned = regexprep(cleaned, '\.\.\.', '');
end

function cleaned = remove_unsupported_escapes(str)
    cleaned = regexprep(str, '\\(?![/"bfnrtu])', '');
end

function cleaned = fix_multiline_text(str)
    cleaned = str;
    
    cleaned = regexprep(cleaned, '\\n', ' ');
    cleaned = regexprep(cleaned, '\n', ' ');
    cleaned = regexprep(cleaned, '\r', ' ');
    
    cleaned = regexprep(cleaned, '"recognized_text"\s*:\s*"[^"]*?(?="\s*[,}])', '"recognized_text": ""');
end

function cleaned = fix_json_quotation(str)
    cleaned = str;
    idx = 1;
    n = length(cleaned);
    
    while idx <= n
        if cleaned(idx) == '"'
            start_idx = idx;
            idx = idx + 1;
            while idx <= n && cleaned(idx) ~= '"'
                if cleaned(idx) == '\' && idx < n && cleaned(idx+1) == '"'
                    idx = idx + 2;
                else
                    idx = idx + 1;
                end
            end
            if idx > n
                cleaned = [cleaned(1:start_idx), '"', cleaned(start_idx+1:end)];
                break;
            end
        end
        idx = idx + 1;
    end
end

function regions_data = parse_json_regex(json_content)
    regions_data = [];
    
    fprintf('    使用备用正则表达式解析方法...\n');
    
    region_blocks = extract_region_blocks(json_content);
    
    if isempty(region_blocks)
        fprintf('    正则表达式未找到任何区域数据\n');
        return;
    end
    
    fprintf('    正则表达式找到 %d 个区域块\n', length(region_blocks));
    
    for i = 1:length(region_blocks)
        block = region_blocks{i};
        
        region = struct();
        
        region_id = extract_field_value(block, 'region_id');
        if ~isempty(region_id)
            region.region_id = str2double(region_id);
        else
            region.region_id = i;
        end
        
        yellow_pixels = extract_field_value(block, 'yellow_pixels');
        if ~isempty(yellow_pixels)
            region.yellow_pixels = str2double(yellow_pixels);
        else
            region.yellow_pixels = 0;
        end
        
        region_pixels_str = extract_array_field(block, 'region_pixels');
        region.region_pixels = parse_coordinate_array(region_pixels_str);
        
        yellow_coords_str = extract_array_field(block, 'yellow_pixels_coords');
        region.yellow_pixels_coords = parse_coordinate_array(yellow_coords_str);
        
        region.recognized_text = '';
        
        fprintf('    区域 %d: %d pixels, %d yellow coords\n', ...
            region.region_id, length(region.region_pixels), length(region.yellow_pixels_coords));
        
        regions_data = [regions_data, region];
    end
end

function blocks = extract_region_blocks(json_str)
    blocks = {};
    
    pattern = '\{\s*"region_id"\s*:[\s\S]*?\}\s*(?=,\s*\{|\s*\])';
    matches = regexp(json_str, pattern, 'match');
    
    if ~isempty(matches)
        blocks = matches;
        return;
    end
    
    pattern = '"region_id":\s*\d+[\s\S]*?(?="region_id":|$)';
    matches = regexp(json_str, pattern, 'match');
    
    if ~isempty(matches)
        blocks = matches;
    end
end

function value = extract_field_value(block, field_name)
    pattern = sprintf('"%s"\\s*:\\s*([^,}\\s]+)', field_name);
    matches = regexp(block, pattern, 'tokens');
    if ~isempty(matches)
        value = matches{1}{1};
    else
        value = '';
    end
end

function arr_str = extract_array_field(block, field_name)
    pattern = sprintf('"%s"\\s*:\\s*\\[([^\\]]*)\\]', field_name);
    matches = regexp(block, pattern, 'tokens');
    if ~isempty(matches)
        arr_str = matches{1}{1};
    else
        arr_str = '';
    end
end

function coords = parse_coordinate_array(arr_str)
    coords = [];
    
    if isempty(arr_str)
        return;
    end
    
    coord_pattern = '\[(\d+)\s*,\s*(\d+)\]';
    matches = regexp(arr_str, coord_pattern, 'tokens');
    
    for i = 1:length(matches)
        x = str2double(matches{i}{1});
        y = str2double(matches{i}{2});
        coords = [coords; x, y];
    end
    
    if isempty(coords)
        num_pattern = '(\d+)';
        nums = regexp(arr_str, num_pattern, 'match');
        nums = cellfun(@str2double, nums);
        if length(nums) >= 2 && mod(length(nums), 2) == 0
            coords = reshape(nums, 2, [])';
        end
    end
end

function new_json_data = transform_json_regions(regions_data, aug_info, orig_height, orig_width)
    new_json_data = struct();
    new_json_data.regions = {};
    
    if isempty(regions_data)
        return;
    end
    
    new_regions = {};
    
    fprintf('    transform_json_regions: %d regions, type=%s\n', length(regions_data), aug_info.type);
    
    for r = 1:length(regions_data)
        region = regions_data(r);
        
        pixels = region.region_pixels;
        fprintf('      region %d: %d pixels, %d yellow coords\n', region.region_id, length(pixels), length(region.yellow_pixels_coords));
        if isempty(pixels)
            fprintf('        skip: empty pixels\n');
            continue;
        end
        
        new_pixels = transform_pixels(pixels, aug_info, orig_height, orig_width);
        
        if isempty(new_pixels)
            fprintf('        transform result empty\n');
            continue;
        end
        
        new_region = struct();
        new_region.region_id = region.region_id;
        new_region.total_pixels = length(new_pixels);
        new_region.recognized_text = region.recognized_text;
        new_region.region_pixels = new_pixels;
        
        if ~isempty(region.yellow_pixels_coords)
            new_yellow = transform_pixels(region.yellow_pixels_coords, aug_info, orig_height, orig_width);
            new_region.yellow_pixels = length(new_yellow);
            new_region.yellow_pixels_coords = new_yellow;
        else
            new_region.yellow_pixels = 0;
            new_region.yellow_pixels_coords = [];
        end
        
        new_regions{end+1} = new_region;
    end
    
    if ~isempty(new_regions)
        new_json_data.regions = new_regions;
    end
end

function new_pixels = transform_pixels(pixels, aug_info, orig_height, orig_width)
    new_pixels = [];
    
    if isempty(pixels)
        return;
    end
    
    pixels = reshape(pixels, [], 2);
    
    fprintf('      transform_pixels: orig %dx%d, type=%s\n', orig_width, orig_height, aug_info.type);
    
    switch aug_info.type
        case 'rotation'
            angle = aug_info.angle;
            center_x = (orig_width + 1) / 2;
            center_y = (orig_height + 1) / 2;
            
            if isfield(aug_info, 'new_width') && isfield(aug_info, 'new_height')
                new_width = aug_info.new_width;
                new_height = aug_info.new_height;
            else
                if angle == 90 || angle == 270
                    new_width = orig_height;
                    new_height = orig_width;
                else
                    new_width = orig_width;
                    new_height = orig_height;
                end
            end
            
            fprintf('      rotation %d: center=(%.1f,%.1f), new_size=%dx%d\n', angle, center_x, center_y, new_width, new_height);
            
            valid_count = 0;
            for i = 1:size(pixels, 1)
                x = pixels(i, 1);
                y = pixels(i, 2);
                
                dx = x - center_x;
                dy = y - center_y;
                
                switch angle
                    case 90
                        new_x = round(center_x - dy);
                        new_y = round(center_y + dx);
                    case 180
                        new_x = round(center_x - dx);
                        new_y = round(center_y - dy);
                    case 270
                        new_x = round(center_x + dy);
                        new_y = round(center_y - dx);
                    otherwise
                        new_x = x;
                        new_y = y;
                end
                
                if new_x >= 1 && new_x <= new_width && new_y >= 1 && new_y <= new_height
                    new_pixels = [new_pixels; new_x, new_y];
                    valid_count = valid_count + 1;
                end
            end
            fprintf('      rotation: %d/%d pixels valid\n', valid_count, size(pixels,1));
            
        case 'crop'
            start_x = aug_info.start_x;
            start_y = aug_info.start_y;
            new_width = aug_info.new_width;
            new_height = aug_info.new_height;
            
            fprintf('      crop: start=(%d,%d), size=%dx%d\n', start_x, start_y, new_width, new_height);
            
            valid_count = 0;
            for i = 1:size(pixels, 1)
                x = pixels(i, 1);
                y = pixels(i, 2);
                
                new_x = x - start_x + 1;
                new_y = y - start_y + 1;
                
                if new_x >= 1 && new_x <= new_width && new_y >= 1 && new_y <= new_height
                    new_pixels = [new_pixels; new_x, new_y];
                    valid_count = valid_count + 1;
                end
            end
            fprintf('      crop: %d/%d pixels valid\n', valid_count, size(pixels,1));
            
        case 'flip'
            direction = aug_info.direction;
            
            valid_count = 0;
            for i = 1:size(pixels, 1)
                x = pixels(i, 1);
                y = pixels(i, 2);
                
                switch direction
                    case 'lr'
                        new_x = orig_width - x + 1;
                        new_y = y;
                    case 'ud'
                        new_x = x;
                        new_y = orig_height - y + 1;
                    otherwise
                        new_x = x;
                        new_y = y;
                end
                
                if new_x >= 1 && new_x <= orig_width && new_y >= 1 && new_y <= orig_height
                    new_pixels = [new_pixels; new_x, new_y];
                    valid_count = valid_count + 1;
                end
            end
            fprintf('      flip %s: %d/%d pixels valid\n', direction, valid_count, size(pixels,1));
            
        otherwise
            new_pixels = pixels;
    end
end

function extract_features_from_augmented(img_folder, json_folder, output_folder)
    json_files = dir(fullfile(json_folder, '*.json'));
    
    if isempty(json_files)
        fprintf('未找到增强JSON文件\n');
        return;
    end
    
    fprintf('找到 %d 个增强JSON文件\n', length(json_files));
    
    for i = 1:length(json_files)
        json_file = fullfile(json_folder, json_files(i).name);
        [~, json_name, ~] = fileparts(json_files(i).name);
        
        fprintf('处理增强文件 %d/%d: %s\n', i, length(json_files), json_files(i).name);
        
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
            fprintf('警告: 未找到对应图片文件: %s\n', img_name);
            continue;
        end
        
        try
            img = imread(img_file);
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
        catch ME
            fprintf('警告: 读取图片失败: %s\n', ME.message);
            continue;
        end
        
        try
            fid = fopen(json_file, 'r');
            content = fread(fid, '*char')';
            fclose(fid);
        catch ME
            fprintf('警告: 读取JSON失败: %s\n', ME.message);
            continue;
        end
        
        features = [];
        content_clean = preclean_json(content);
        try
            json_data = jsondecode(content_clean);
            if ~isfield(json_data, 'regions') || isempty(json_data.regions)
                fprintf('文件 %s 未找到区域信息，跳过\n', json_files(i).name);
                continue;
            end
            regions_data = json_data.regions;
        catch ME
            fprintf('警告: JSON解析失败: %s，跳过\n', ME.message);
            continue;
        end
        
        fprintf('  找到 %d 个区域\n', length(regions_data));
        
        for j = 1:length(regions_data)
            reg = regions_data(j);
            
            if ~isfield(reg, 'region_pixels') || isempty(reg.region_pixels)
                fprintf('  区域 %d 没有像素数据，跳过\n', reg.region_id);
                continue;
            end
            
            coords = reg.region_pixels;
            if size(coords, 2) ~= 2
                fprintf('  区域 %d 像素格式错误，跳过\n', reg.region_id);
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
            
            recognized_text = '';
            if isfield(reg, 'recognized_text')
                recognized_text = reg.recognized_text;
            end
            
            feature = struct('region_id', reg.region_id, 'recognized_text', recognized_text, ...
                'position', struct('min_x', min_x, 'max_x', max_x, 'min_y', min_y, 'max_y', max_y, 'width', width, 'height', height), ...
                'center', struct('x', center_x, 'y', center_y), ...
                'axes', struct('major_axis', major_axis, 'minor_axis', minor_axis), ...
                'roughness', roughness, 'complexity', complexity, ...
                'gray_level', struct('mean', mean_gray, 'std', std_gray));
            features = [features, feature];
        end
        
        if ~isempty(features)
            output_data = struct('regions', features);
            json_output = jsonencode(output_data, 'PrettyPrint', true);
            output_file = fullfile(output_folder, [json_name, '.json']);
            fid = fopen(output_file, 'w');
            fwrite(fid, json_output);
            fclose(fid);
            fprintf('  特征提取完成，保存到 %s\n', output_file);
        end
    end
    
    fprintf('特征提取完成！\n');
end

function stop = myProgressCallback(info, iterations_per_epoch)
    persistent epoch_count;
    if info.Initialization
        epoch_count = 0;
        fprintf('\n[Epoch 0] 初始化完成，开始训练...\n');
        stop = false;
        return;
    end

    if info.IterationDone
        current_iteration = info.Iteration;
        current_epoch = floor(current_iteration / iterations_per_epoch) + 1;
        if current_epoch > epoch_count
            epoch_count = current_epoch;
            if current_epoch <= 10
                fprintf('\n====================================\n');
                fprintf('[Epoch %d/10] 开始训练...\n', current_epoch);
                fprintf('====================================\n');
            end
        end

        if mod(current_iteration, iterations_per_epoch) == 0 || current_iteration == 1
            loss_str = 'N/A';
            if ~isempty(info.TrainingLoss) && ~isnan(info.TrainingLoss)
                loss_str = sprintf('%.6f', info.TrainingLoss);
            end
            val_loss_str = 'N/A';
            if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss) && ~isnan(info.ValidationLoss)
                val_loss_str = sprintf('%.6f', info.ValidationLoss);
            end
            fprintf('[Epoch %d] Iteration %d | Loss: %s | Val Loss: %s\n', ...
                current_epoch, current_iteration, loss_str, val_loss_str);
        end
    end

    if info.EpochDone
        current_epoch = info.EpochIndex;
        fprintf('\n----- Epoch %d 完成 -----\n', current_epoch);
    end

    stop = false;
end

function train_model(features_all_path, features_pre_path, mode_path, data_path)
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
        image_name = strrep(base_name, 'region_analysis__', '');

        img_file = fullfile(data_path, 'change', [image_name, '.tif']);
        if ~exist(img_file, 'file')
            fprintf('警告：找不到图片 %s\n', img_file);
            continue;
        end

        img = imread(img_file);
        img_resized = imresize(img, [256, 256]);
        if size(img_resized, 3) == 3
            img_gray = rgb2gray(img_resized);
        else
            img_gray = img_resized;
        end
        img_normalized = double(img_gray) / 255;

        try
            json_data = jsondecode(fileread(json_file_path));
            regions = json_data.regions;
        catch
            fprintf('警告：无法解析JSON文件 %s\n', json_file_path);
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

    iterations_per_epoch = ceil(train_size / 8);

    progressFcn = @(info) myProgressCallback(info, iterations_per_epoch);

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
            'MaxEpochs', 10, ...
            'MiniBatchSize', 8, ...
            'Verbose', true, ...
            'Plots', 'none');

        fprintf('====================================\n');
        fprintf('开始训练模型...\n');
        fprintf('训练参数: MaxEpochs=%d, MiniBatchSize=%d\n', 10, 8);
        fprintf('====================================\n');
        net = trainNetwork(train_images, train_features, layers, options);
        fprintf('模型训练成功\n');

    catch ME
        fprintf('模型训练失败: %s\n', ME.message);
        diary off;
        error('模型训练失败');
    end
    
    model_file = fullfile(mode_path, 'representation_learning_model.mat');
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
        
        pred_file = fullfile(features_pre_path, sprintf('prediction_%d.json', i));
        json_str = jsonencode(pred_result);
        fid = fopen(pred_file, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
    end
    
    fprintf('预测结果保存完成\n');
end
