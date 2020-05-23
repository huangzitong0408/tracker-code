function img = preprocess(img)
[r,c] = size(img);
win = window2(r,c,@hann);
save win
eps = 1e-5;
img = log(double(img)+1);
img = (img-mean(img(:)))/(std(img(:))+eps); % 【把数据进行标准高斯正态化的方法】
img = img.*win;  % 【一般物体不在边缘，加窗进一步弱化周围背景？】理解相关背后数值运算关系
end


% 有正有负的矩阵图像与目标进行相关操作，不相关的区域会趋近于0,有利于弱化背景的影响，避免多个峰值
% 因为若数据都为正数，相关操作对应相乘再相加仍会得到较大的值

% 通常都会对特征提取的图像信息进行一些预处理：
% （1）用log函数对像素值进行处理，降低对比度（contrasting lighting situation）。
% （2）进行平均值为0，范数为1的归一化。
% （3）用余弦窗口进行滤波，降低边缘的像素值。