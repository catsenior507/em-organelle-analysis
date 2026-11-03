function code_dir = project_assert_cwd(mode)
%PROJECT_ASSERT_CWD 检查当前工作目录是否为 code 目录，否则报错（默认）或告警
%
%   code_dir = project_assert_cwd()          % pwd 不是 code 目录 -> error
%   code_dir = project_assert_cwd('warn')    % pwd 不是 code 目录 -> warning
%
%   为什么需要它：历史脚本用 `../data/...` 这类 pwd 相对路径，并且用
%   `system('python ocr_scale_bar.py')` 这类相对命令调用外部脚本。在错误的
%   目录里运行不会抛异常，只会**静默**读错数据或调用失败——这类错误在批量
%   运行里极难发现。宁可吵，也不要错。
%
%   注意：能改用 project_paths() 得到绝对路径的脚本应优先改路径；本函数是给
%   仍然依赖 pwd 的脚本兜底用的。

if nargin < 1 || isempty(mode)
    mode = 'error';
end

code_dir = fileparts(mfilename('fullpath'));
cwd = pwd;

% 用"该目录下是否存在本文件"判断，避免大小写、短路径(8.3)、软链接等写法差异造成误报
in_code_dir = (exist(fullfile(cwd, 'project_assert_cwd.m'), 'file') == 2) || ...
              strcmpi(strtrim(cwd), strtrim(code_dir));
if in_code_dir
    return;
end

msg = sprintf(['当前工作目录不是 code 目录：\n' ...
    '  当前 pwd : %s\n' ...
    '  期望目录 : %s\n' ...
    '历史脚本使用 pwd 相对路径（如 ''../data/...''）和相对命令（python ocr_scale_bar.py），\n' ...
    '在其它目录运行会静默读错数据。请先 cd 到 code 目录，或改用 project_paths() 的绝对路径。'], ...
    cwd, code_dir);

switch lower(mode)
    case 'warn'
        warning('project_assert_cwd:notCodeDir', '%s', msg);
    otherwise
        error('project_assert_cwd:notCodeDir', '%s', msg);
end
end
