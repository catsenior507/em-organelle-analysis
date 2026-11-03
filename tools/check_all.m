function check_all()
%CHECK_ALL  Syntax-validate the key files under code/ and print project_paths().
%
%   Usage:
%       matlab -batch "addpath('...\tools'); check_all"
%
%   Notes on the checks used (learned the hard way on R2025b):
%     * mtree(file)  -- constructing the tree PARSES the file and throws on a
%       syntax error.  This is the real syntax test.  The old mtree API method
%       .any() no longer exists in R2025b, so do NOT rely on it.
%     * checkcode(file,'-m0') still returns style/performance messages, so
%       numel(r)==0 is NOT a syntax test.  Its messages are reported
%       separately and explicitly labelled as advisory.
%     * checkcode needs a resolvable file path (a bare function name fails with
%       "Cannot open file" when the working directory is elsewhere).
%
%   ASCII-only on purpose: MATLAB parses .m files with the system code page
%   (GBK here), so a UTF-8 .m with Chinese comments fails to load.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
codedir = fullfile(root, 'code');
addpath(codedir);

files = {'project_paths', 'project_assert_cwd', 'analyze_emergent_entropy', ...
         'find_scale_bar', 'normalize_features_um', 'simple_region_features', ...
         'extract_region_features', 'analyze_annotated_regions'};

fprintf('==== syntax validity (mtree parse) + advisory message counts ====\n');
syntax_fail = 0;
for i = 1:numel(files)
    name = files{i};
    fpath = fullfile(codedir, [name '.m']);

    if ~exist(fpath, 'file')
        fprintf('%-28s MISSING FILE: %s\n', name, fpath);
        syntax_fail = syntax_fail + 1;
        continue;
    end

    % --- real syntax test: parse the file ---
    try
        mtree(fpath);
        syntax = 'OK';
    catch ME
        syntax = 'SYNTAX ERROR';
        syntax_fail = syntax_fail + 1;
        fprintf('%-28s %s: %s\n', name, syntax, ME.message);
        continue;
    end

    % --- advisory messages, reported separately ---
    try
        r = checkcode(fpath, '-m0');
        nmsg = numel(r);
    catch
        nmsg = -1;
    end
    fprintf('%-28s syntax=%-6s advisory_messages=%d\n', name, syntax, nmsg);
end
fprintf('------------------------------------------------------------------\n');
fprintf('SYNTAX FAILURES = %d\n\n', syntax_fail);

fprintf('==== project_paths() runtime resolution ====\n');
try
    P = project_paths();
    fprintf('root                : %s\n', P.root);
    fprintf('annotations_v2      : %d (exists)\n', exist(P.annotations_v2, 'dir') > 0);
    fprintf('regions_v2          : %d (exists)\n', exist(P.regions_v2, 'dir') > 0);
    fprintf('dataset_json        : %d (exists)\n', exist(P.dataset_json, 'file') > 0);
    fprintf('features_json       : %d json\n', numel(dir(fullfile(P.features_json, '*.json'))));
    fprintf('normalized_features : %d csv\n', numel(dir(fullfile(P.normalized_features, '*.csv'))));
    fprintf('scale_mat           : %d mat\n', numel(dir(fullfile(P.scale_mat, '*.mat'))));
    fprintf('python (OCR-capable): %s\n', P.python);
    if isempty(P.python) || strcmp(P.python, 'python')
        fprintf('  [WARN] no OCR-capable python resolved\n');
    end
    if exist(P.python, 'file')
        [st, out] = system(['"' P.python '" -c "import PIL, pytesseract; print(''ocr deps ok'')"']);
        fprintf('python probe        : status=%d out=%s\n', st, strtrim(out));
    end
catch ME
    fprintf('project_paths FAILED: %s\n', ME.message);
    syntax_fail = syntax_fail + 1;
end

fprintf('\n==== done, syntax failures = %d ====\n', syntax_fail);
end
