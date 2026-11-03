
% 测试JSON解析修复
fprintf('====================================\n');
fprintf('开始测试JSON解析修复...\n');
fprintf('====================================\n');

base_path = pwd;
data_path = fullfile(base_path, '..', 'data', 'features_json');

if ~exist(data_path, 'dir')
    fprintf('错误：找不到features_json文件夹\n');
    return;
end

json_files = dir(fullfile(data_path, '*.json'));

if isempty(json_files)
    fprintf('未找到JSON文件\n');
    return;
end

fprintf('找到 %d 个JSON文件\n', length(json_files));

success_count = 0;
regex_count = 0;

for i = 1:min(5, length(json_files))
    json_file = fullfile(data_path, json_files(i).name);
    fprintf('\n------------------------------------\n');
    fprintf('测试文件 %d: %s\n', i, json_files(i).name);
    
    fid = fopen(json_file, 'r');
    if fid == -1
        fprintf('无法打开文件\n');
        continue;
    end
    json_content = fread(fid, '*char')';
    fclose(fid);
    
    fprintf('文件大小: %d 字符\n', length(json_content));
    
    regions_data = parse_json_regions(json_content);
    
    if ~isempty(regions_data)
        success_count = success_count + 1;
        fprintf('✅ 成功解析 %d 个区域\n', length(regions_data));
        
        if isfield(regions_data, 'region_pixels') && ~isempty(regions_data(1).region_pixels)
            fprintf('   区域 %d 有 %d 个像素点\n', regions_data(1).region_id, length(regions_data(1).region_pixels));
        end
    else
        fprintf('❌ 解析失败\n');
    end
end

fprintf('\n====================================\n');
fprintf('测试完成！\n');
fprintf('成功解析 %d/%d 个测试文件\n', success_count, min(5, length(json_files)));
fprintf('====================================\n');
