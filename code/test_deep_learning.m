% 测试深度学习工具箱是否正常工作

% 清除工作空间和命令窗口
clear all;
clc;

% 开启日记记录
diary('test_deep_learning.log');

fprintf('====================================\n');
fprintf('测试深度学习工具箱\n');
fprintf('====================================\n');
fprintf('开始测试时间: %s\n', datestr(now));

% 检查深度学习工具箱是否安装
if exist('trainNetwork', 'file')
    fprintf('✓ trainNetwork函数存在\n');
else
    fprintf('✗ trainNetwork函数不存在\n');
    error('深度学习工具箱未安装');
end

% 检查其他关键函数是否存在
functions_to_check = {'fullyConnectedLayer', 'reluLayer', 'regressionLayer', 'trainingOptions'};
for i = 1:length(functions_to_check)
    func_name = functions_to_check{i};
    if exist(func_name, 'file')
        fprintf('✓ %s函数存在\n', func_name);
    else
        fprintf('✗ %s函数不存在\n', func_name);
    end
end

% 创建简单的测试数据
fprintf('\n创建测试数据...\n');
% 生成随机输入数据 (100个样本，每个样本10个特征)
train_images = rand(100, 10);
% 生成随机目标数据 (100个样本，每个样本2个输出)
train_features = rand(100, 2);
% 生成测试数据
test_images = rand(20, 10);
test_features = rand(20, 2);

fprintf('训练数据大小: %d x %d\n', size(train_images, 1), size(train_images, 2));
fprintf('训练目标大小: %d x %d\n', size(train_features, 1), size(train_features, 2));

% 创建神经网络模型
fprintf('\n创建神经网络模型...\n');
% 使用更简单的网络结构
layers = [
    featureInputLayer(10)
    fullyConnectedLayer(2)
    regressionLayer
];

% 显示网络结构
fprintf('网络结构:\n');
disp(layers);

% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 10, ...
    'ValidationData', {test_images, test_features}, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'none');

% 训练模型
fprintf('\n训练神经网络模型...\n');
try
    net = trainNetwork(train_images, train_features, layers, options);
    fprintf('✓ 模型训练成功\n');
    fprintf('模型类型: %s\n', class(net));
catch ME
    fprintf('✗ 模型训练失败: %s\n', ME.message);
    error('模型训练失败');
end

% 测试模型
fprintf('\n测试模型...\n');
try
    predicted_features = predict(net, test_images);
    fprintf('✓ 模型预测成功\n');
    fprintf('预测结果大小: %d x %d\n', size(predicted_features, 1), size(predicted_features, 2));
    
    % 计算误差
    mse = mean((predicted_features - test_features).^2, 'all');
    rmse = sqrt(mse);
    fprintf('均方误差 (MSE): %.4f\n', mse);
    fprintf('均方根误差 (RMSE): %.4f\n', rmse);
catch ME
    fprintf('✗ 模型预测失败: %s\n', ME.message);
    error('模型预测失败');
end

fprintf('\n====================================\n');
fprintf('测试完成！深度学习工具箱工作正常\n');
fprintf('====================================\n');
fprintf('结束测试时间: %s\n', datestr(now));

% 关闭日记记录
diary off;