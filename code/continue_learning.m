
% 继续学习脚本 - 从特征提取阶段开始
clear all;
close all;
clc;

diary('continue_learning.log');
diary on;

fprintf('====================================\n');
fprintf('继续表征学习 - 特征提取和训练\n');
fprintf('====================================\n');

base_path = pwd;
data_path = fullfile(base_path, '..', 'data');
change_img_path = fullfile(data_path, 'change');
change_feature_json_path = fullfile(data_path, 'change_feature_json');
change_features_all_path = fullfile(data_path, 'change_features_all');
change_features_pre_path = fullfile(data_path, 'change_features_pre');
change_mode_path = fullfile(base_path, '..', 'change_mode');

fprintf('检查文件...\n');
if ~exist(change_feature_json_path, 'dir')
    fprintf('错误: 未找到增强特征JSON文件夹\n');
    diary off;
    return;
end

json_files = dir(fullfile(change_feature_json_path, '*.json'));
fprintf('找到 %d 个增强JSON文件\n', length(json_files));

fprintf('\n开始提取增强图片的特征...\n');
extract_features_from_augmented(change_img_path, change_feature_json_path, change_features_all_path);

fprintf('\n开始训练模型...\n');
train_model(change_features_all_path, change_features_pre_path, change_mode_path, data_path);

fprintf('\n====================================\n');
fprintf('继续学习完成！\n');
fprintf('====================================\n');

diary off;
