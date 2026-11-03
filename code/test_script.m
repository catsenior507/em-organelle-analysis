% 测试脚本
clear all;
close all;
clc;

fprintf('测试脚本开始运行...\n');

% 测试路径设置
base_path = pwd;
fprintf('当前工作目录: %s\n', base_path);

data_path = fullfile(base_path, '..', 'data');
fprintf('数据目录: %s\n', data_path);

% 测试文件夹存在性
if exist(data_path, 'dir')
    fprintf('数据目录存在\n');
else
    fprintf('数据目录不存在\n');
end

% 测试features_all文件夹
features_all_path = fullfile(data_path, 'features_all');
if exist(features_all_path, 'dir')
    fprintf('features_all文件夹存在\n');
    % 列出其中的文件
    files = dir(fullfile(features_all_path, '*.json'));
    fprintf('找到 %d 个JSON文件\n', length(files));
    if length(files) > 0
        fprintf('第一个文件: %s\n', files(1).name);
    end
else
    fprintf('features_all文件夹不存在\n');
end

% 测试创建文件夹
mode_path = fullfile(base_path, '..', 'mode');
if ~exist(mode_path, 'dir')
    fprintf('创建mode文件夹...\n');
    mkdir(mode_path);
    if exist(mode_path, 'dir')
        fprintf('mode文件夹创建成功\n');
    else
        fprintf('mode文件夹创建失败\n');
    end
else
    fprintf('mode文件夹已存在\n');
end

fprintf('测试脚本运行完成\n');
