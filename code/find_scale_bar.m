% 从红色标注图像中识别比例尺

% 路径全部按本文件位置解析（旧代码用 '../data/...'，依赖 pwd，易静默读错目录）
addpath(fileparts(mfilename('fullpath')));
P = project_paths();
% 本脚本仍会以相对命令调用 python ocr_scale_bar.py，pwd 不对时给出告警而非静默失败
project_assert_cwd('warn');

% 获取所有以 red 开头的 tif 文件
data_dir = P.separated;
red_files = dir(fullfile(data_dir, 'red*.tif'));

% 确保保存结果的目录存在
save_dir = P.scale_mat;
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 处理每个红色标注图像
for i = 1:length(red_files)
    % 获取当前文件路径
    red_image_path = fullfile(data_dir, red_files(i).name);
    disp(['处理文件: ', red_image_path]);
    
    % 读取红色标注图像
    img = imread(red_image_path);
    
    % 提取红色像素（237:28:36）
    red_pixels = (img(:, :, 1) == 237) & (img(:, :, 2) == 28) & (img(:, :, 3) == 36);
    
    % 连通区域分析
    cc = bwconncomp(red_pixels);
    num_regions = cc.NumObjects;
    
    % 提取每个区域的属性
    region_props = regionprops(cc, 'BoundingBox', 'Area', 'PixelList');
    
    % 存储每个区域的横向长度和位置
    region_widths = zeros(num_regions, 1);
    region_heights = zeros(num_regions, 1);
    region_boxes = zeros(num_regions, 4);
    
    for j = 1:num_regions
        bbox = region_props(j).BoundingBox;
        region_widths(j) = bbox(3);
        region_heights(j) = bbox(4);
        region_boxes(j, :) = bbox;
    end
    
    % 找到横向长度最长的区域（主比例尺）
    [max_width, main_scale_idx] = max(region_widths);
    
    % 移除主比例尺，处理剩余区域
    remaining_regions = setdiff(1:num_regions, main_scale_idx);
    
    % 对剩余区域进行 OCR
    scale_value = 0;
    scale_unit = '';
    
    if ~isempty(remaining_regions)
        % 创建一个包含所有剩余红色像素的图像
        remaining_pixels = false(size(red_pixels));
        for j = remaining_regions
            pixel_list = region_props(j).PixelList;
            remaining_pixels(sub2ind(size(red_pixels), pixel_list(:, 2), pixel_list(:, 1))) = true;
        end
        
        % 尝试使用 Matlab 的 OCR
        try
            % 将二值图像转换为灰度图像
            gray_img = uint8(255 * double(remaining_pixels));
            
            % 进行 OCR
            ocr_results = ocr(gray_img);
            text = ocr_results.Text;
            
            disp(['OCR 识别结果: ', text]);
            
            % 提取数字和单位
            % 寻找类似 "2um"、"10μm"、"500nm" 或 "1mm" 的模式
            tokens = regexp(text, '([0-9.]+)\s*(nm|mm|[uμ]?m)', 'tokens');
            if ~isempty(tokens)
                scale_value = str2double(tokens{1}{1});
                scale_unit = tokens{1}{2};
                disp(['识别到比例尺值: ', num2str(scale_value), scale_unit]);
            else
                disp('OCR 未识别到有效的比例尺值');
            end
        catch ME
            disp(['Matlab OCR 失败: ', ME.message]);
            disp('将尝试使用 Python OCR');
            
            % 保存剩余区域为临时图像（放在 code\temp，不随 pwd 漂移）
            temp_dir = fullfile(P.code, 'temp');
            if ~exist(temp_dir, 'dir')
                mkdir(temp_dir);
            end
            imwrite(gray_img, fullfile(temp_dir, 'temp_scale_mark.tif'));
            
            % 调用 Python OCR 脚本（用绝对路径 + project_paths 探测到的解释器，
            % 旧代码 'python ocr_scale_bar.py' 依赖 pwd 和 PATH，失败时无声无息）
            python_cmd = sprintf('"%s" "%s"', P.python, P.ocr_scale_script);
            [status, result] = system(python_cmd);
            if status == 0
                disp(['Python OCR 结果: ', result]);
                % 解析结果
            tokens = regexp(result, '([0-9.]+)\s*(nm|mm|[uμ]?m)', 'tokens');
            if ~isempty(tokens)
                scale_value = str2double(tokens{1}{1});
                scale_unit = tokens{1}{2};
                disp(['Python OCR 识别到比例尺值: ', num2str(scale_value), scale_unit]);
            end
            else
                disp('Python OCR 也失败了');
            end
        end
    else
        disp('未找到其他红色区域，无法识别比例尺值');
    end
    
    % 计算像素到微米的转换比例
    scale_ok = false;   % 标定是否可信；下游（如 normalize_features_um.m）必须检查该字段
    if scale_value > 0 && ~isempty(max_width) && max_width > 0
        % 单位换算：将不同单位转换为微米(um)
        switch lower(scale_unit)
            case 'nm'
                % 纳米转换为微米
                scale_value_um = scale_value / 1000;
                disp(['单位换算: ', num2str(scale_value), 'nm = ', num2str(scale_value_um), 'um']);
            case 'mm'
                % 毫米转换为微米
                scale_value_um = scale_value * 1000;
                disp(['单位换算: ', num2str(scale_value), 'mm = ', num2str(scale_value_um), 'um']);
            case {'um', 'μm', 'u', 'μ'}
                % 已经是微米单位
                scale_value_um = scale_value;
            otherwise
                % 单位无法识别：旧实现静默按微米处理，可能造成 10^6 量级误差
                scale_value_um = scale_value;
                warning('find_scale_bar:unknownUnit', ...
                    ['无法识别的比例尺单位 "%s"（文件 %s），已按微米(um)处理。\n' ...
                     '若单位实际是 mm 或 nm，结果会差 10^3~10^6 倍，请人工核对 data\\scale_mat 下的标定。'], ...
                    scale_unit, red_files(i).name);
        end

        pixel_to_um = scale_value_um / max_width;
        scale_ok = true;
        disp(['像素到微米的转换比例: ', num2str(pixel_to_um), ' um/像素']);
    else
        % 旧实现在这里静默代入 pixel_to_um = 0.1，误差可达 20–100 倍，
        % 而且下游完全看不出来。现在改为：不编造数值，写 NaN + scale_ok=false，
        % 并大声告警，让调用方有机会拒绝该标定。
        pixel_to_um = NaN;
        warning('find_scale_bar:scaleFailed', ...
            ['比例尺标定失败，已放弃标定：%s\n' ...
             '  原因：未能识别比例尺数值或主比例尺宽度无效 (scale_value=%s, max_width=%s)\n' ...
             '  结果：pixel_to_um = NaN, scale_ok = false（不再使用旧的默认值 0.1）\n' ...
             '  下游脚本应拒绝该标定，或改用人工标定 / tools/morpho/extract_regions.py 的 scale_report.json。'], ...
            red_files(i).name, num2str(scale_value), num2str(max_width));
    end
    
    % 生成保存文件名（去掉 red 前缀，前加 scale_）
    filename = red_files(i).name;
    % 旧实现 strrep(filename,'red','') 会删除文件名中**所有**出现的 "red"，
    % 这里只剥离开头的 red 前缀，其余部分原样保留。
    % 注意：'red_11_0002.tif' 去掉前缀后剩下 '_11_0002'，拼接得到 'scale__11_0002.mat'。
    % 双下划线是历史约定：data\scale_mat 下已有文件以及 tools/morpho/extract_regions.py
    % 的 scale__<name>.mat 查找都依赖它，不要"顺手"改成单下划线。
    filename_without_red = regexprep(filename, '^red', '', 'once');
    save_filename = ['scale_', filename_without_red(1:end-4), '.mat'];
    save_path = fullfile(save_dir, save_filename);
    
    % 保存转换比例（新增 scale_ok：false 表示本次标定不可信）
    save(save_path, 'pixel_to_um', 'max_width', 'scale_value', 'scale_unit', 'scale_ok');
    disp(['结果保存到: ', save_path]);
    
    % 显示最终结果
    disp('比例尺识别完成');
    disp(['主比例尺宽度: ', num2str(max_width), ' 像素']);
    disp(['识别到的长度值: ', num2str(scale_value), scale_unit]);
    disp(['转换比例: ', num2str(pixel_to_um), ' um/像素 (scale_ok=', num2str(scale_ok), ')']);
    disp('--------------------------------------------------');
end