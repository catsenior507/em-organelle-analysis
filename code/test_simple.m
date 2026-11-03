
% 简单测试脚本 - 直接测试JSON解析
clc; clear; close all;

fprintf('====================================\n');
fprintf('简单JSON解析测试\n');
fprintf('====================================\n');

% 测试文件路径
json_path = fullfile(pwd, '..', 'data', 'features_json', 'region_analysis__11_0002.json');

if ~exist(json_path, 'file')
    fprintf('找不到测试文件\n');
    return;
end

% 读取文件
fprintf('读取JSON文件...\n');
fid = fopen(json_path, 'r');
json_content = fread(fid, '*char')';
fclose(fid);

fprintf('文件大小: %d 字符\n', length(json_content));

% 尝试直接使用jsondecode
fprintf('\n尝试直接jsondecode...\n');
try
    data = jsondecode(json_content);
    fprintf('✅ 直接解析成功！\n');
    if isfield(data, 'regions')
        fprintf('   发现 %d 个区域\n', length(data.regions));
    end
catch ME
    fprintf('❌ 直接解析失败: %s\n', ME.message);
    
    fprintf('\n尝试清理后解析...\n');
    try
        cleaned = preclean_json_test(json_content);
        data = jsondecode(cleaned);
        fprintf('✅ 清理后解析成功！\n');
        if isfield(data, 'regions')
            fprintf('   发现 %d 个区域\n', length(data.regions));
        end
    catch ME2
        fprintf('❌ 清理后仍失败: %s\n', ME2.message);
        
        fprintf('\n尝试备用正则表达式方法...\n');
        regions = parse_json_regex_test(json_content);
        if ~isempty(regions)
            fprintf('✅ 正则表达式解析成功！\n');
            fprintf('   发现 %d 个区域\n', length(regions));
        else
            fprintf('❌ 所有方法都失败\n');
        end
    end
end

fprintf('\n====================================\n');
fprintf('测试完成\n');
fprintf('====================================\n');

function json_content_clean = preclean_json_test(json_content)
    json_content_clean = json_content;
    
    json_content_clean = regexprep(json_content_clean, '\r\n', '\n');
    json_content_clean = regexprep(json_content_clean, '\r', '\n');
    
    json_content_clean = remove_unsupported_escapes_test(json_content_clean);
    json_content_clean = fix_multiline_text_test(json_content_clean);
    json_content_clean = fix_json_quotation_test(json_content_clean);
    json_content_clean = strtrim(json_content_clean);
end

function cleaned = remove_unsupported_escapes_test(str)
    cleaned = regexprep(str, '\\(?![/"bfnrtu])', '');
end

function cleaned = fix_multiline_text_test(str)
    cleaned = str;
    cleaned = regexprep(cleaned, '\\n', ' ');
    cleaned = regexprep(cleaned, '\n', ' ');
end

function cleaned = fix_json_quotation_test(str)
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

function regions_data = parse_json_regex_test(json_content)
    regions_data = [];
    
    region_blocks = extract_region_blocks_test(json_content);
    
    if isempty(region_blocks)
        return;
    end
    
    for i = 1:length(region_blocks)
        block = region_blocks{i};
        
        region = struct();
        
        region_id = extract_field_value_test(block, 'region_id');
        if ~isempty(region_id)
            region.region_id = str2double(region_id);
        else
            region.region_id = i;
        end
        
        yellow_pixels = extract_field_value_test(block, 'yellow_pixels');
        if ~isempty(yellow_pixels)
            region.yellow_pixels = str2double(yellow_pixels);
        else
            region.yellow_pixels = 0;
        end
        
        region_pixels_str = extract_array_field_test(block, 'region_pixels');
        region.region_pixels = parse_coordinate_array_test(region_pixels_str);
        
        yellow_coords_str = extract_array_field_test(block, 'yellow_pixels_coords');
        region.yellow_pixels_coords = parse_coordinate_array_test(yellow_coords_str);
        
        region.recognized_text = '';
        
        regions_data = [regions_data, region];
    end
end

function blocks = extract_region_blocks_test(json_str)
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

function value = extract_field_value_test(block, field_name)
    pattern = sprintf('"%s"\\s*:\\s*([^,}\\s]+)', field_name);
    matches = regexp(block, pattern, 'tokens');
    if ~isempty(matches)
        value = matches{1}{1};
    else
        value = '';
    end
end

function arr_str = extract_array_field_test(block, field_name)
    pattern = sprintf('"%s"\\s*:\\s*\\[([^\\]]*)\\]', field_name);
    matches = regexp(block, pattern, 'tokens');
    if ~isempty(matches)
        arr_str = matches{1}{1};
    else
        arr_str = '';
    end
end

function coords = parse_coordinate_array_test(arr_str)
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
