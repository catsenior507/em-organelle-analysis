@echo off
cd /d "d:\dachuang\code"
echo 开始运行测试脚本...
matlab -nodisplay -nosplash -r "try, run('test_script.m'), catch ME, disp(['Error: ' ME.message]), end, exit;" > test_output.txt
echo 测试脚本运行完成，输出已保存到 test_output.txt
type test_output.txt
pause
