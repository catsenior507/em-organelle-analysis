% ==========================================================================
% 【已被取代 SUPERSEDED】分析分离的标注区域并进行文字识别
% 读取绿色像素图片，识别被绿色圈起的区域
% 检查这些区域是否包含黄色像素并进行文字识别
%
% ⚠ 本脚本是旧的区域提取流水线，已被 tools/morpho/extract_regions.py 取代。
%   需要正确特征时请运行：
%       python tools/morpho/extract_regions.py --separated "data/分离的标注信息" ^
%           --originals "data/原图片" --scale-dir "data/scale_mat" --out "data/annotations_v2"
%   旧实现的三个根本缺陷（新实现已修正，本脚本未重写）：
%     1. 分割反了：对绿色掩码取反（~green_mask）后做连通域，得到的是**背景**；
%        绿色像素是手绘的细胞器轮廓线，真正的区域是轮廓内部（fill_holes(green)）。
%     2. 区域像素被截断到 1000 点，并在 JSON 中写入字面量 `...`，
%        导致 91/94 个输出文件不是合法 JSON。
%     3. 坐标语义是 MATLAB find 的 [row, col]，下游却按 (x, y)=(列, 行) 读，x/y 互换。
%   本次只做了两处低成本修补（Python 解释器路径探测、截断不再产生非法 JSON），
%   其余逻辑保持原样以便与历史结果对照；不要再基于它的输出训练模型。
% ==========================================================================

% 定义基本路径（按本文件位置解析，不再依赖 pwd）
addpath(fileparts(mfilename('fullpath')));
P = project_paths();

base_dir = P.separated;
output_dir = P.features_json;

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 递归获取所有TIFF文件
all_tiff_files = rdir(base_dir);

% 分离绿色和黄色文件
green_files = {};
yellow_files = {};

for i = 1:length(all_tiff_files)
    file_path = all_tiff_files{i};
    [~, name, ~] = fileparts(file_path);
    
    % 精确匹配前缀
    if startsWith(name, 'green')
        green_files{end+1} = file_path;
    elseif startsWith(name, 'yellow')
        yellow_files{end+1} = file_path;
    end
end

% 确保绿色和黄色文件数量匹配
if length(green_files) ~= length(yellow_files)
    disp('警告: 绿色和黄色文件数量不匹配');
end

% 处理每对文件
for i = 1:min(length(green_files), length(yellow_files))
    green_path = green_files{i};
    yellow_path = yellow_files{i};
    
    % 提取文件名（不含扩展名）
    [~, green_name, ~] = fileparts(green_path);
    [~, yellow_name, ~] = fileparts(yellow_path);
    
    % 确保文件名前缀匹配
    if ~startsWith(green_name, 'green') || ~startsWith(yellow_name, 'yellow')
        continue;
    end
    
    % 提取共同部分（假设文件名格式为 green_xxx.tif 和 yellow_xxx.tif）
    green_suffix = green_name(6:end);  % 去除 'green' 前缀
    yellow_suffix = yellow_name(7:end);  % 去除 'yellow' 前缀
    
    if strcmp(green_suffix, yellow_suffix)
        disp(['处理文件对: ' green_name ' 和 ' yellow_name]);
        
        % 读取图像
        try
            green_im = imread(green_path);
            yellow_im = imread(yellow_path);
        catch ME
            disp(['读取文件失败: ' ME.message]);
            continue;
        end
        
        % 确保图像是二值图像（只保留绿色像素）
        green_mask = any(green_im > 0, 3);
        
        % 识别所有连通的非绿色像素区域
        non_green_mask = ~green_mask;
        
        % 使用bwlabel直接识别连通区域
        [labeled_regions, num_regions] = bwlabel(non_green_mask);
        
        % 边界掩码仍然是绿色像素区域
        boundary_mask = green_mask;
        
        disp(['识别到 ' num2str(num_regions) ' 个连通的非绿色像素区域']);
        
        % 准备保存结果的文件
        output_file = fullfile(output_dir, sprintf('region_analysis_%s.json', green_suffix));
        fid = fopen(output_file, 'w');
        
        % 开始写入JSON
        fprintf(fid, '{\n');
        fprintf(fid, '  "regions": [\n');

        % 已实际写出的区域对象数：用于决定逗号，避免"最后一个区域没有黄色像素"
        % 时留下尾随逗号（那同样是非法 JSON）
        emitted_regions = 0;
        % 单区域最多写出的坐标点数（与旧实现一致，但不再写 `...`）
        max_emit_points = 1000;
        truncated_any = false;
        
        % 分析每个区域
        for j = 1:num_regions
            % 提取当前区域
            region_mask = (labeled_regions == j);
            
            % 检查区域是否包含黄色像素
            yellow_mask = any(yellow_im > 0, 3);
            yellow_in_region = any(region_mask & yellow_mask, 'all');
            
            if yellow_in_region
                % 提取区域内的所有像素
                [region_rows, region_cols] = find(region_mask);
                
                % 提取区域内的黄色像素
                yellow_region_pixels = find(region_mask & yellow_mask);
                [yellow_rows, yellow_cols] = ind2sub(size(yellow_mask), yellow_region_pixels);
                
                % 尝试进行文字识别
                recognized_text = '';
                try
                    if ~isempty(yellow_rows) && ~isempty(yellow_cols)
                        % 计算边界框
                        rmin = min(yellow_rows);
                        rmax = max(yellow_rows);
                        cmin = min(yellow_cols);
                        cmax = max(yellow_cols);
                        
                        % 扩展边界以包含更多上下文（增加边框大小）
                        margin = 20; % 进一步增加边框大小以保留更多上下文
                        rmin = max(1, rmin - margin);
                        rmax = min(size(yellow_im, 1), rmax + margin);
                        cmin = max(1, cmin - margin);
                        cmax = min(size(yellow_im, 2), cmax + margin);
                        
                        % 裁剪图像到边界框
                        cropped_im = yellow_im(rmin:rmax, cmin:cmax, :);
                        
                        % 转换为灰度并二值化
                        gray_temp = rgb2gray(cropped_im);
                        % 增强对比度
                        gray_temp = imadjust(gray_temp);
                        % 使用全局阈值二值化，更适合文字识别
                        binary_temp = imbinarize(gray_temp, 'global');
                        % 去除小的噪声点
                        binary_temp = bwareaopen(binary_temp, 2);
                        % 膨胀操作，增强文字连通性
                        se = strel('disk', 1);
                        binary_temp = imdilate(binary_temp, se);
                        
                        if any(binary_temp(:))
                            % 保存处理后的图像用于调试
                            debug_dir = fullfile(output_dir, 'debug');
                            if ~exist(debug_dir, 'dir')
                                mkdir(debug_dir);
                            end
                            debug_file = fullfile(debug_dir, sprintf('debug_ocr_%s_%d.tif', green_suffix, j));
                            imwrite(binary_temp, debug_file);
                            
                            % 进行OCR，调用Python脚本
                            try
                                % 保存二值化图像用于OCR
                                temp_image = fullfile(debug_dir, sprintf('temp_ocr_%s_%d.tif', green_suffix, j));
                                imwrite(binary_temp, temp_image);
                                
                                % 构建Python命令：解释器与脚本都用绝对路径。
                                % 旧代码写死 '"D:\dachuang\.venv\Scripts\python.exe" ocr.py'，
                                % 该路径已不存在（项目已移动），且 ocr.py 依赖 cwd。
                                python_cmd = sprintf('"%s" "%s" "%s"', ...
                                    P.python, fullfile(P.code, 'ocr.py'), temp_image);
                                
                                % 执行Python命令
                                [status, result] = system(python_cmd);
                                
                                if status == 0
                                    % 解析JSON结果
                                    result_struct = jsondecode(result);
                                    if result_struct.success
                                        recognized_text = result_struct.text;
                                    else
                                        recognized_text = ['识别失败: ' result_struct.error];
                                    end
                                else
                                    recognized_text = ['识别失败: 执行Python脚本失败'];
                                end
                                
                                % 调试信息
                                disp(['Region ' num2str(j) ' OCR result: "' recognized_text '"']);
                            catch ME
                                recognized_text = ['识别失败: ' ME.message];
                                disp(['Region ' num2str(j) ' OCR error: ' ME.message]);
                            end
                            

                        end
                    end
                catch ME
                    recognized_text = ['识别失败: ' ME.message];
                end
                
                % 构建JSON对象（区域之间用逗号分隔，由 emitted_regions 决定，避免尾随逗号）
                if emitted_regions > 0
                    fprintf(fid, ',\n');
                end
                fprintf(fid, '    {\n');
                fprintf(fid, '      "region_id": %d,\n', j);
                fprintf(fid, '      "total_pixels": %d,\n', length(region_rows));
                fprintf(fid, '      "yellow_pixels": %d,\n', length(yellow_region_pixels));
                fprintf(fid, '      "pixel_note": "region_pixels/yellow_pixels_coords 仅为前 %d 个点，非完整区域；正确实现见 tools/morpho/extract_regions.py",\n', max_emit_points);
                
                % 输出区域像素坐标：截断但**不写 `...`**，并显式给出实际写出的点数，
                % 这样 JSON 始终可解析（旧实现写死字面量 ,... 使文件非法）
                n_region_emit = min(length(region_rows), max_emit_points);
                truncated_any = truncated_any || (length(region_rows) > n_region_emit);
                fprintf(fid, '      "region_pixels_emitted": %d,\n', n_region_emit);
                fprintf(fid, '      "region_pixels": [');
                for k = 1:n_region_emit
                    if k > 1
                        fprintf(fid, ',');
                    end
                    fprintf(fid, '[%d,%d]', region_rows(k), region_cols(k));
                end
                fprintf(fid, '],\n');
                
                % 输出黄色像素坐标（同样截断但不写 `...`）
                n_yellow_emit = min(length(yellow_rows), max_emit_points);
                truncated_any = truncated_any || (length(yellow_rows) > n_yellow_emit);
                fprintf(fid, '      "yellow_pixels_coords_emitted": %d,\n', n_yellow_emit);
                fprintf(fid, '      "yellow_pixels_coords": [');
                for k = 1:n_yellow_emit
                    if k > 1
                        fprintf(fid, ',');
                    end
                    fprintf(fid, '[%d,%d]', yellow_rows(k), yellow_cols(k));
                end
                fprintf(fid, '],\n');
                
                % 输出识别到的文字（转义 反斜杠/引号/换行，否则 OCR 结果或错误信息
                % 里的 Windows 路径会把 JSON 弄坏——同样是"文件必须可解析"的一部分）
                bs = char(92);   % 反斜杠
                text_json = strrep(recognized_text, bs, [bs bs]);
                text_json = strrep(text_json, '"', '\"');
                text_json = regexprep(text_json, '[\r\n\t]', ' ');
                fprintf(fid, '      "recognized_text": "%s"\n', text_json);
                emitted_regions = emitted_regions + 1;
                
                % 结束当前区域的JSON对象（逗号已在下一个对象写之前统一处理）
                fprintf(fid, '    }');
            end
        end
        
        % 结束JSON
        fprintf(fid, '\n  ]\n');
        fprintf(fid, '}\n');
        
        fclose(fid);
        disp(['分析完成，结果已保存到: ' output_file]);
        if truncated_any
            % 截断是真实的信息损失，必须让它可见，而不是像旧实现那样埋一个 `...`
            warning('analyze_annotated_regions:truncatedPixels', ...
                ['%s 中至少一个区域的像素坐标被截断到前 %d 个点' ...
                 '（JSON 仍然合法，但 region_pixels 不是完整区域）。\n' ...
                 '需要完整形状请使用 tools/morpho/extract_regions.py 生成的 data\\annotations_v2（轮廓为简化多边形）。'], ...
                output_file, max_emit_points);
        end

    end
end

% 递归目录搜索函数
function file_paths = rdir(base_dir)
    % 初始化结果
    file_paths = {};
    
    % 搜索当前目录下的所有TIFF文件
    current_files = dir(fullfile(base_dir, '*.tif'));
    for i = 1:length(current_files)
        file_paths{end+1} = fullfile(base_dir, current_files(i).name);
    end
    
    % 搜索子目录
    subdirs = dir(base_dir);
    subdirs = subdirs([subdirs.isdir]); % 只保留目录
    
    for i = 1:length(subdirs)
        subdir_name = subdirs(i).name;
        if ~strcmp(subdir_name, '.') && ~strcmp(subdir_name, '..')
            subdir_path = fullfile(base_dir, subdir_name);
            % 递归搜索子目录
            subfiles = rdir(subdir_path);
            file_paths = [file_paths; subfiles];
        end
    end
end