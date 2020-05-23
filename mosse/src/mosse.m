clc,clear,close all;
%get images from source directory

%此处仅仅用于得到图片序列所在地址
datadir = 'F:/研一研二研三/研一下/machine_learning/目标追踪经典算法MOSSE/mosse-tracker-master/data/';
dataset = 'Surfer';
path = [datadir dataset];
img_path = [path '/img/'];
D = dir([img_path, '*.jpg']);%在img_path下的所有jpg后缀文件的地址放入D中 【dir函数得到文件结构数组，查看变量空间可知】

seq_len = length(D(not([D.isdir])));%得到图片总数 【每张图片对应一个结构体】
if exist([img_path num2str(1, '%04i.jpg')], 'file')
    img_files = num2str((1:seq_len)', [img_path '%04i.jpg']); % 【学习：得到每帧图像的路径，含文件名】
else
    error('No image files found in the directory.');
end

% select target from first frame
im = imread(img_files(1,:)); % 【读取首帧图像】
f = figure('Name', 'Select object to track'); 
imshow(im);
rect = getrect; % 【图像切割函数】
close(f); clear f;
center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];

% plot gaussian
sigma = 100;
gsize = size(im);%获取图像尺寸【彩色图像】
[R,C] = ndgrid(1:gsize(1), 1:gsize(2));  
%两个矩阵R和C，都是m行n列
g = gaussC(R,C, sigma, center);%通过R和C产生size等同于im的高斯滤波函数 【高斯的峰值为所选目标的中心位置】
g = mat2gray(g); %【实现图像矩阵的归一化】
mesh(g);
% randomly warp original image to create training set
if (size(im,3) == 3) 
    img = rgb2gray(im); 
end
%%
img = imcrop(img, rect); %【图像切割函数，给定区域rect切割】
g = imcrop(g, rect); 
G = fft2(g); % 【由H=G/F来推导滤波器H，这里高斯G即为理想输出，即希望目标中心附近响应为高斯形状，并在目标有峰值】
figure;
%%
mesh(g); %【对比裁剪前后高斯图形，裁剪后峰值在中间】
figure;
mesh(abs(G)); % 【注意时域高斯则频域也为高斯，但频域的峰值在零频率处】
%将高斯滤波函数变换到频域
height = size(g,1);
width = size(g,2);
%% 这里给定滤波器的初始值，当前滤波器的产生与前一帧的滤波器有关
fi = preprocess(imresize(img, [height width]));%imresize(img, [height width])将图片调整成滤波器的大小 【这里错了，应该是调整成和高斯响应尺寸应相同】
Ai = (G.*conj(fft2(fi))); % 【conj函数：用于计算复数的共轭值   高斯与所选目标相关操作？--时域相关=频域相乘(一个取共轭)】 
Bi = (fft2(fi).*conj(fft2(fi)));

%% 以上用所选目标产生了一个高斯函数以及得到一个初始的滤波器，这个滤波器是理想化的，由一个样本得到，因此会过拟合；
%% 下面为了去除这种过拟合，需要更多用于训练滤波器的样本（下面用了旋转）
figure;
imshow(img);
N = 128;
for i = 1:N
    fi = preprocess(rand_warp(img)); % 【rand_warp对图像进行随机旋转，用于对Ai和Bi进行修正】
    Ai = Ai + (G.*conj(fft2(fi)));  
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
    %【功能查看的技巧：1、把图片显示出来；2、在变量空间进行显示】
    %figure;
    %imshow(fi);
end

% MOSSE online training regimen
eta = 0.25;
fig = figure('Name', 'MOSSE');
t = figure;
mkdir(['results_' dataset]);
for i = 1:size(img_files, 1)   % 【循环处理每张图片】
    img = imread(img_files(i,:)); % 【读取第i张图片】
    im = img;  % 【多余？选中查看后面是否有使用该变量】
    if (size(img,3) == 3)  % 【注意参数3的使用】
        img = rgb2gray(img);
    end
    if (i == 1)
        Ai = eta.*Ai;
        Bi = eta.*Bi;
    %%  第二帧开始训练更新模型---注意下面fi的不同（前后帧的截取）
    else  % 【注意这里Hi、fi、gi的size都与rect相同】
        Hi = Ai./Bi;
%         imshow(abs(ifft2(Hi)),[]); %【？？？？？】
%         mesh(abs(Hi));  %【频域较难直观看出效果】
        fi = imcrop(img, rect);
%         imshow(fi);
%         mesh(fi);
         fi = preprocess(imresize(fi, [height width]));   % 【学习：鼠标停在对应变量能查看变量值；这里fi尺寸已经是height*width】
%         imshow(fi);
%         mesh(fi);  % 【可以看出处理后的图像数值有正有负，且被映射为标准高斯正态化】
        gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi)))); % 【先归一化，再乘255，线性映射成0~255】
        % mesh(gi); %【good debug】
%         mesh(abs(Hi.*fft2(fi)));
        maxval = max(gi(:))   % 【学习：找矩阵的最大值】响应最大的位置即为目标位置
        [P, Q] = find(gi == maxval);  % 【找到峰值位置，即相关峰值？；P为横坐标，Q为纵坐标】这里可能有多个峰值位置
        dx = mean(P)-height/2;
        dy = mean(Q)-width/2;
      
        rect = [rect(1)+dy rect(2)+dx width height];  % 【更新裁剪框的位置】
        fi = imcrop(img, rect); 
        fi = preprocess(imresize(fi, [height width]));
        Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;  % 更新 【question:为什么这里的G一直都是用最开始的那个】
        Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;
%% the answer of question:由于我们希望在每帧目标中心的响应是以该点为中心的高斯形状，而fi已经是新的目标框内容，其对应的响应G的高斯峰值仍然应在目标框的中心
    end
    % visualization
    text_str = ['Frame: ' num2str(i)];
    box_color = 'green';
    position=[1 1];
    result = insertText(im, position,text_str,'FontSize',15,'BoxColor',...
                     box_color,'BoxOpacity',0.4,'TextColor','white');
    result = insertShape(result, 'Rectangle', rect, 'LineWidth', 3);
    imwrite(result, ['results_' dataset num2str(i, '/%04i.jpg')]);
	imshow(result);
    drawnow;
    rect
end
