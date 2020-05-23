function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014


	%if the target is large, lower the resolution, we don't need that much
	%detail.the result of prod([a b]) is a*b.
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold ��prodΪ��ˡ�
	if resize_image
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end

    a = [1 1 1;1 -8 1;1 1 1];

	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));  %����������һ֡��ȥѰ��Ŀ���λ�ã�ֻ����һС��ΧȥѰ�ң��ӿ��ٶȡ�

% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size; % ��Ϊ���ø�˹������target_sz��ƥ�䣬��target_sz�ܴ����˹��̫С����ܲ��ܵó���Ҫ�Ľ����
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size))); % ������mesh�����鿴��
    %% floor(window_sz / cell_size) �ó�window���ж��ٸ�cell_size ���ǵ������ռ�Ҳ��
    %% ��yf�����ǽ���Ԥ���Ժ�õ�����������������������Ϊ���ģ�������Ϊ����ѵ������ģ����Ŀ�괦���Ϊ����õ�����״����yf�ĸ�˹��״��
	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	

 	if show_visualization  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
 	end
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
%% ��ʼ�ӵ�һ֡��������֮ǰ����Ԥ������
	for frame = 1:numel(img_files)
		%load image
		im = imread([video_path img_files{frame}]);
        im0 = im;
%         if frame > 1
%            im0 = imread([video_path img_files{frame-1}]);
%         end
        
		if size(im,3) || size(im0,3) > 1
			im = rgb2gray(im);
%           im0 = rgb2gray(im0);
		end
		if resize_image
			im = imresize(im, 0.5);
%            im0 = imresize(im0, 0.5);
		end

		tic()  % tic -> toc ����ʱ��

		if frame > 1
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
            %and the size of temp whitch is feature is unchanged:50*21*31

			patch = get_subwindow(im, pos, window_sz);
            temp = get_features(patch, features, cell_size, cos_window);
			zf = fft2(temp);

			%calculate response of the classifier at all shifts
			switch kernel.type
			case 'gaussian'
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			case 'polynomial'
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear'
				kzf = linear_correlation(zf, model_xf);
			end
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection FD 2.6
            
			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta, horiz_delta] = find(response == max(response(:)),1); % ��ѧϰ��ʹ��һ��max������������ֵ��
			% ��ע�⣬������������Ӧ������ֱ����ı仯��
            if vert_delta > size(zf,1) / 2  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
        end %end of (if frame>1)

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
%         figure;
%         subplot(121); imshow(im);
%         subplot(122);imshow(patch);
		xf = fft2(get_features(patch, features, cell_size, cos_window));%save xf;  ��f��β�Ķ���ʾ����Ҷ�任��

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian'
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial'
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear'
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training,FD 2.8

		if frame == 1 %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;  % ���ڶ�֡�ս���ʱ���㷨�е�z�����׶Σ����������ٳɹ��󣬻�ȡĿ�����Ϊxf����ģ�ͽ��и��¡�
		else
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;  % ��interp_factorΪѧϰ���ӡ�
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();

		%visualization
		if show_visualization
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image
		positions = positions * 2;
	end
end

