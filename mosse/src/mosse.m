clc,clear,close all;
%get images from source directory

%�˴��������ڵõ�ͼƬ�������ڵ�ַ
datadir = 'F:/��һ�ж�����/��һ��/machine_learning/Ŀ��׷�پ����㷨MOSSE/mosse-tracker-master/data/';
dataset = 'Surfer';
path = [datadir dataset];
img_path = [path '/img/'];
D = dir([img_path, '*.jpg']);%��img_path�µ�����jpg��׺�ļ��ĵ�ַ����D�� ��dir�����õ��ļ��ṹ���飬�鿴�����ռ��֪��

seq_len = length(D(not([D.isdir])));%�õ�ͼƬ���� ��ÿ��ͼƬ��Ӧһ���ṹ�塿
if exist([img_path num2str(1, '%04i.jpg')], 'file')
    img_files = num2str((1:seq_len)', [img_path '%04i.jpg']); % ��ѧϰ���õ�ÿ֡ͼ���·�������ļ�����
else
    error('No image files found in the directory.');
end

% select target from first frame
im = imread(img_files(1,:)); % ����ȡ��֡ͼ��
f = figure('Name', 'Select object to track'); 
imshow(im);
rect = getrect; % ��ͼ���и����
close(f); clear f;
center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];

% plot gaussian
sigma = 100;
gsize = size(im);%��ȡͼ��ߴ硾��ɫͼ��
[R,C] = ndgrid(1:gsize(1), 1:gsize(2));  
%��������R��C������m��n��
g = gaussC(R,C, sigma, center);%ͨ��R��C����size��ͬ��im�ĸ�˹�˲����� ����˹�ķ�ֵΪ��ѡĿ�������λ�á�
g = mat2gray(g); %��ʵ��ͼ�����Ĺ�һ����
mesh(g);
% randomly warp original image to create training set
if (size(im,3) == 3) 
    img = rgb2gray(im); 
end
%%
img = imcrop(img, rect); %��ͼ���и������������rect�и
g = imcrop(g, rect); 
G = fft2(g); % ����H=G/F���Ƶ��˲���H�������˹G��Ϊ�����������ϣ��Ŀ�����ĸ�����ӦΪ��˹��״������Ŀ���з�ֵ��
figure;
%%
mesh(g); %���ԱȲü�ǰ���˹ͼ�Σ��ü����ֵ���м䡿
figure;
mesh(abs(G)); % ��ע��ʱ���˹��Ƶ��ҲΪ��˹����Ƶ��ķ�ֵ����Ƶ�ʴ���
%����˹�˲������任��Ƶ��
height = size(g,1);
width = size(g,2);
%% ��������˲����ĳ�ʼֵ����ǰ�˲����Ĳ�����ǰһ֡���˲����й�
fi = preprocess(imresize(img, [height width]));%imresize(img, [height width])��ͼƬ�������˲����Ĵ�С ��������ˣ�Ӧ���ǵ����ɺ͸�˹��Ӧ�ߴ�Ӧ��ͬ��
Ai = (G.*conj(fft2(fi))); % ��conj���������ڼ��㸴���Ĺ���ֵ   ��˹����ѡĿ����ز�����--ʱ�����=Ƶ�����(һ��ȡ����)�� 
Bi = (fft2(fi).*conj(fft2(fi)));

%% ��������ѡĿ�������һ����˹�����Լ��õ�һ����ʼ���˲���������˲��������뻯�ģ���һ�������õ�����˻����ϣ�
%% ����Ϊ��ȥ�����ֹ���ϣ���Ҫ��������ѵ���˲���������������������ת��
figure;
imshow(img);
N = 128;
for i = 1:N
    fi = preprocess(rand_warp(img)); % ��rand_warp��ͼ����������ת�����ڶ�Ai��Bi����������
    Ai = Ai + (G.*conj(fft2(fi)));  
    Bi = Bi + (fft2(fi).*conj(fft2(fi)));
    %�����ܲ鿴�ļ��ɣ�1����ͼƬ��ʾ������2���ڱ����ռ������ʾ��
    %figure;
    %imshow(fi);
end

% MOSSE online training regimen
eta = 0.25;
fig = figure('Name', 'MOSSE');
t = figure;
mkdir(['results_' dataset]);
for i = 1:size(img_files, 1)   % ��ѭ������ÿ��ͼƬ��
    img = imread(img_files(i,:)); % ����ȡ��i��ͼƬ��
    im = img;  % �����ࣿѡ�в鿴�����Ƿ���ʹ�øñ�����
    if (size(img,3) == 3)  % ��ע�����3��ʹ�á�
        img = rgb2gray(img);
    end
    if (i == 1)
        Ai = eta.*Ai;
        Bi = eta.*Bi;
    %%  �ڶ�֡��ʼѵ������ģ��---ע������fi�Ĳ�ͬ��ǰ��֡�Ľ�ȡ��
    else  % ��ע������Hi��fi��gi��size����rect��ͬ��
        Hi = Ai./Bi;
%         imshow(abs(ifft2(Hi)),[]); %��������������
%         mesh(abs(Hi));  %��Ƶ�����ֱ�ۿ���Ч����
        fi = imcrop(img, rect);
%         imshow(fi);
%         mesh(fi);
         fi = preprocess(imresize(fi, [height width]));   % ��ѧϰ�����ͣ�ڶ�Ӧ�����ܲ鿴����ֵ������fi�ߴ��Ѿ���height*width��
%         imshow(fi);
%         mesh(fi);  % �����Կ���������ͼ����ֵ�����и����ұ�ӳ��Ϊ��׼��˹��̬����
        gi = uint8(255*mat2gray(ifft2(Hi.*fft2(fi)))); % ���ȹ�һ�����ٳ�255������ӳ���0~255��
        % mesh(gi); %��good debug��
%         mesh(abs(Hi.*fft2(fi)));
        maxval = max(gi(:))   % ��ѧϰ���Ҿ�������ֵ����Ӧ����λ�ü�ΪĿ��λ��
        [P, Q] = find(gi == maxval);  % ���ҵ���ֵλ�ã�����ط�ֵ����PΪ�����꣬QΪ�����꡿��������ж����ֵλ��
        dx = mean(P)-height/2;
        dy = mean(Q)-width/2;
      
        rect = [rect(1)+dy rect(2)+dx width height];  % �����²ü����λ�á�
        fi = imcrop(img, rect); 
        fi = preprocess(imresize(fi, [height width]));
        Ai = eta.*(G.*conj(fft2(fi))) + (1-eta).*Ai;  % ���� ��question:Ϊʲô�����Gһֱ�������ʼ���Ǹ���
        Bi = eta.*(fft2(fi).*conj(fft2(fi))) + (1-eta).*Bi;
%% the answer of question:��������ϣ����ÿ֡Ŀ�����ĵ���Ӧ���Ըõ�Ϊ���ĵĸ�˹��״����fi�Ѿ����µ�Ŀ������ݣ����Ӧ����ӦG�ĸ�˹��ֵ��ȻӦ��Ŀ��������
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
