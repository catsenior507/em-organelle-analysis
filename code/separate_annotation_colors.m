% 分离标注图片中的特定颜色像素
% 绿色 (0, 180, 75) - 细胞器轮廓
% 黄色 (255, 242, 0) - 细胞器名字
% 红色 (237, 28, 36) - 比例尺

% 设置输入和输出目录
input_dir = '../data/标注图片';
output_dir = '../data/分离的标注信息';

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    disp(['创建输出目录: ' output_dir]);
end

% 获取输入目录中的所有图片文件
image_files = dir(fullfile(input_dir, '*.tif'));
image_files = [image_files; dir(fullfile(input_dir, '*.jpg'))];
image_files = [image_files; dir(fullfile(input_dir, '*.png'))];

% 检查是否有图片文件
if isempty(image_files)
    disp('输入目录中没有找到图片文件！');
    return;
end

disp(['找到 ' num2str(length(image_files)) ' 个图片文件']);

% 定义颜色阈值
colors = struct();
colors.green = [0, 180, 75];
colors.yellow = [255, 242, 0];
colors.red = [237, 28, 36];

% 定义颜色英文名
color_names = struct();
color_names.green = 'green';
color_names.yellow = 'yellow';
color_names.red = 'red';

% 处理每个图片文件
for i = 1:length(image_files)
    % 获取当前图片路径
    filename = image_files(i).name;
    annotation_path = fullfile(input_dir, filename);
    
    disp(['\n处理文件: ' filename]);
    
    % 读取标注图片
    [im, map] = imread(annotation_path);
    
    % 显示图像信息
    disp(['原始图像大小: ' num2str(size(im))]);
    disp(['是否有颜色映射: ' num2str(~isempty(map))]);
    
    % 确保图像是RGB格式
    if size(im, 3) ~= 3
        disp('图像不是RGB格式，尝试转换...');
        if ~isempty(map)
            % 索引图像转换为RGB
            im = ind2rgb(im, map);
            im = im2uint8(im);
        else
            % 灰度图像转换为RGB
            im = cat(3, im, im, im);
        end
    end
    
    % 再次显示转换后的图像信息
    disp(['转换后图像大小: ' num2str(size(im))]);
    
    % 确保图像只有3个通道
    if size(im, 3) > 3
        disp('图像通道数过多，只保留前3个通道...');
        im = im(:, :, 1:3);
    end
    
    % 对每种颜色进行分离
    for color = fieldnames(colors)'
        color_name = color{1};
        target_color = colors.(color_name);
        color_english_name = color_names.(color_name);
        
        % 创建掩码，匹配目标颜色
        mask = (im(:,:,1) == target_color(1)) & ...
               (im(:,:,2) == target_color(2)) & ...
               (im(:,:,3) == target_color(3));
        
        % 创建分离后的图像
        separated_im = zeros(size(im), 'uint8');
        
        % 确保im是uint8类型
        if ~isa(im, 'uint8')
            im = im2uint8(im);
        end
        
        % 使用逻辑掩码进行赋值
        for j = 1:3
            separated_im(:,:,j) = uint8(mask) .* im(:,:,j);
        end
        
        % 生成输出文件名
        [~, base_name, ~] = fileparts(annotation_path);
        output_filename = [color_english_name '_' base_name '.tif'];
        output_path = fullfile(output_dir, output_filename);
        
        % 保存分离后的图像
        imwrite(separated_im, output_path);
        disp(['已保存 ' color_english_name ' 标注到: ' output_path]);
    end
end

disp('\n所有图片颜色分离完成！');
disp(['共处理 ' num2str(length(image_files)) ' 个图片文件']);