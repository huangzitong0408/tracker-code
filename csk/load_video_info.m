function [img_files, pos, target_sz, resize_image, ground_truth, video_path] = load_video_info(video_path)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), whether to resize the video to half
%   (boolean), and the ground truth information for precision calculations
%   (Nx2, for N frames). The ordering of coordinates is always [y, x].
%
%   The path to the video is returned, since it may change if the images
%   are located in a sub-folder (as is the default for MILTrack's videos).
%
%   Jo?o F. Henriques, 2012
%   http://www.isr.uc.pt/~henriques/

	%load ground truth from text file (MILTrack's format)
	text_files = dir([video_path '*_rect.txt']); %ground_truth_rect.txt�а���ÿһ֡��Ŀ����ʵ����λ��
	assert(~isempty(text_files), 'No initial position and groundtruth_rect.txt to load.')%���Ժ�������������������������������Ϣ
                                        %�磺assert(a>=0,'���aС��0')����aֵ���ڵ���0�����������������У���aֵС��0�����ӡ������ʾ��Ϣ"���aС��0"

                                            %textscan��������ʵ������textread��࣬����������˸���Ĳ��������˺ܶ����ơ�
                                            %textscan���ʺ϶�����ļ������Դ��ļ����κ�λ�ÿ�ʼ���룬��textread ֻ�ܴ��ļ���ͷ��ʼ���룻
                                            %ֻ����һ��ϸ�����󣬶�textreadҪ���ض�����飻�ṩ����ת���������ݵ�ѡ���ṩ���û���������ò�����
	f = fopen([video_path text_files(1).name]);
	ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
    ground_truth = cat(2, ground_truth{:});%cat�������Ӿ���2��ʾ�����ӵľ���Ϊ��ά��������������Ϊ��Ҫ���Ӿ��������������
	fclose(f);                              
                                            
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];%Ŀ�������[height, width]
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);%posΪ��һ֡��Ŀ���������ĵ�λ��[y,x]��floor���������ȡ��,
                                                                      %y��x�Ƕ�����ͼƬ��˵������ֵ
	
	%interpolate missing annotations, and store positions instead of boxes
    %����ȱ�ٵ�ע�ͣ�ͬʱ����λ�ö����Ǻ��ӡ�
	try
		ground_truth = interp1(1 : 5 : size(ground_truth,1), ground_truth(1:5:end,:), 1:size(ground_truth,1));
                                     %��ֵyi=interp1(x,y,xi,'method'),����x��yΪ��ֵ�㣬yiΪ�ڱ���ֵ��xi���Ĳ�ֵ�����x,yΪ������methodȱʡΪ���Բ�ֵ
                                     %size(A,1)ȡA��������size(A,2)ȡA��������size(A)ȡA��������
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2; %ground_truthΪ����֡��Ŀ���������ĵ�λ������[y, x]
	catch  %, wrong format or we just don't have ground truth data.
		ground_truth = [];
	end
	
	%list all frames. first, try MILTrack's format, where the initial and
	%final frame numbers are stored in a text file. if it doesn't work,
	%try to load all jpg/jpg files in the folder.
%% �ٷ�Դ����� list all frames	
	text_files = dir([video_path '*_frames.txt']);%*_frame.txt�ǵ�һ֡ͼ���������������һ֡ͼ��������š�
	if ~isempty(text_files)
		f = fopen([video_path text_files(1).name]);
		frames = textscan(f, '%f,%f');
		fclose(f);
		
		%see if they are in the 'imgs' subfolder or not
		if exist([video_path num2str(frames{1}, 'img/%04i.jpg')], 'file')
			video_path = [video_path 'img/'];
		elseif ~exist([video_path num2str(frames{1}, 'img/%04i.jpg')], 'file')
			error('No image files to load.')
		end
		
		%list the files
		img_files = num2str((frames{1} : frames{2})', '%04i.jpg');
		img_files = cellstr(img_files);
	else
		%no text file, just list all images
		img_files = dir([video_path '*.jpg']); %�ڵ�ǰfolder��Ѱ��jpgͼƬ
		if isempty(img_files)
            video_path = [video_path 'img/']; %���ڵ�ǰfolder��û��jpgͼƬ�����뵱ǰfolder��subfolder��img��Ѱ��jpgͼƬ
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.') %����img���subfolder��Ҳû��jpg�ļ�������������Ϣ
		end
		img_files = sort({img_files.name});%sort(A)��A�������������л�����������Ĭ�϶��Ƕ�A�����������С�sort(A,'descend')�ǽ�������
	end
%% �Լ�д���ĵ�list all frames��ͨ���Բ���ٷ�����
%     startFrame = 1; %��ͼ����Ų��Ǵ�0001��ʼ(BlurCar1���ݼ�������������0247.jpg��ʼ)��ֻ�轫startFrame��Ϊ��ʼ֡����(��BlurCarΪstartFrame = 247)
%     endFrame = length(ground_truth) + startFrame - 1;
%
%     if exist([video_path num2str(startFrame, 'img/%04i.jpg')], 'file')
%         video_path = [video_path 'img/'];
%     elseif ~exist([video_path num2str(startFrame, 'img/%04i.jpg')], 'file')
%         error('No image files to load.')
%     end	
% 
%     %list the files
%     img_files = num2str((startFrame : endFrame)', '%04i.jpg');
%     img_files = cellstr(img_files);
%%
    %if the target is too large, use a lower resolution - no need for so
    %���ͼƬ̫��ʹ�ýϵͷֱ��� - ����Ҫ����(���������е����ݼ���ͼ�񶼺�С)
	%much detail
	if sqrt(prod(target_sz)) >= 100 %prod(target_sz)����target_sz�����ĸ�Ԫ�صĳ˻�����target_szΪ��һ֡ͼ����Ŀ������ĵ�[height, width]
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
		resize_image = true;
	else
		resize_image = false;
	end
end

