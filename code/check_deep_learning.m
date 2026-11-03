% 检查深度学习工具箱函数是否存在

% 清除工作空间和命令窗口
clear all;
clc;

fprintf('====================================\n');
fprintf('检查深度学习工具箱函数\n');
fprintf('====================================\n');

% 检查关键函数是否存在
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

% 检查深度学习工具箱是否安装
fprintf('\n检查深度学习工具箱是否安装...\n');
try
    % 尝试创建一个简单的网络层
    layer = featureInputLayer(10);
    fprintf('✓ 深度学习工具箱安装成功\n');
    fprintf('创建的层类型: %s\n', class(layer));
except ME
    fprintf('✗ 深度学习工具箱安装失败: %s\n', ME.message);
end

fprintf('\n====================================\n');
fprintf('检查完成！\n');
fprintf('====================================\n');