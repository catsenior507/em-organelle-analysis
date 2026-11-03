% 运行测试脚本并显示输出

% 开启日记记录
diary('test_deep_learning.log');

% 运行测试脚本
run('test_deep_learning.m');

% 关闭日记记录
diary off;

% 显示日志内容
type('test_deep_learning.log');