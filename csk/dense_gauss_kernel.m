function k = dense_gauss_kernel(sigma, x, y)
%DENSE_GAUSS_KERNEL Gaussian Kernel with dense sampling.
%   Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
%   between input images X and Y, which must both be MxN. They must also
%   be periodic (ie., pre-processed with a cosine window). The result is
%   an MxN map of responses.
%
%   If X and Y are the same, ommit the third parameter to re-use some
%   values, which is faster.
%
%   Jo?o F. Henriques, 2012
%   http://www.isr.uc.pt/~henriques/

%Ϊʲô˵ѭ������Ϳ���ʵ����һ��ͼƬ�϶������������ڵ������Ӵ��ڽ����ܼ������أ�
%�ҵ�����ǣ�ѭ��λ���൱��Ŀ���(��һ֡�ó�����Ŀ�����ķ�Χ��Ҳ�������Ϊ��ķ�Χ)���������ڵ�ǰ֡��ȡ��ͼƬ(Ҳ������������)����ÿ�ƶ�һ�ξͿ������
%Ŀ��������������ص����ֵĻ����ֵ��C(x)y�����������ֵԽ�󣬾�˵��Խ�п�����Ŀ������λ�ã������Ϳ��Եó�Ŀ�������������ÿһ���ֵĻ����ֵ��

	xf = fft2(x);  %x in Fourier domain
                   %ͨ�����mesh(x)��mesh(xf)���Կ���x����������������λ��(���Ѿ����й������Ҵ�����)����xf���ź�������������ϵ��������ĸ����ϡ�
                   %�����ɶ�ά����Ҷ�������ʾ����ģ����任����ԭ���������ģ���Ƶ���������зֲ��ڱ任ϵ�������
                   %���ĸ���(ͼ����Ӱ��)�������õĶ�ά����Ҷ�任�����ԭ���������Ͻǣ���ôͼ���ź�������������ϵ������
                   %���ĸ����ϡ�������ͼ���������е�Ƶ����
                   %��MATLAB��fft�������и���Ҷ�任ʱ����任����ԭ��������Ͻǣ����������γɵĸ���Ҷ���ڵ�ͼ�������������Ľ�
                   %��͵����˸���Ҷ�任��Ƶ��ͼ�ϵĸ�����ͼ���ϸ��㲢������һһ��Ӧ�Ĺ�ϵ�����Ҫ�����Ӧ����(MATLAB�ٷ�˵������Ƶ�����Ƶ�Ƶ������)��
                   %MATLAB�ٷ���������ʹ��fftshift����������������������ʹ����circshift(ifft2(xyf),floor(size(x)/2))������ʵ����Ҳ�ǽ���Ƶ�����Ƶ�Ƶ�����ġ�
	xx = x(:)' * x(:);  %squared norm of x :����x(:)Ϊ��x���н�����ϳ�һ����
		
	if nargin >= 3  %general case, x and y are different ��nargin�������ж�������������ĺ���
		yf = fft2(y);
		yy = y(:)' * y(:);
	else
		%auto-correlation of x, avoid repeating a few operations
		yf = xf;
		yy = xx;
	end

	%cross-correlation term in Fourier domain
	xyf = xf .* conj(yf); %������Eq.4
% 	xy = real(circshift(ifft2(xyf), floor(size(x)/2)));  %to spatial domain
                               %circshiftѭ����λ�ĺ���,ͬʱ�Ծ�������к��е���λ����circshift(A,[col, row])������col��ʾ��λ�ƣ�row��ʾ��λ��
                              
			       %������Eq.4��֪C(u)v=$F^{-1}$($F^*(u)$$\bigodot$F(v)),Ҳ����˵C(y)x=ifft(F(xy))=ifft(F(x) .* F*(y)),
			       %���Կ���ֻҪִ��ifft(F(x) .* F*(y))�����Ƕ�y������ѭ�����������ͬʱ���y����ѭ��������x�Ļ�����ԣ�
			       %��ΪC(y)x�ǽ�y����ѭ����������x�Ļ�����ԡ�
    xy = real(fftshift(ifft2(xyf)));%MATLAB�ٷ��ĵ������Ľ���Ƶ�����Ƶ�Ƶ�����ĵĺ�����fftshift,����ĵ�circshift(ifft2(xyf), floor(size(x)/2))ʵ�ֵ�
                                    %����Ҳ�ǽ���Ƶ�����Ƶ�Ƶ�����ģ�ֻ��fftshift������ִ�п��������� X �Ǿ����� fftshift �Ὣ X �ĵ�һ������
                                    %�������޽��������ڶ�������������޽���.)����circshift������һλһλ���ƶ���ֱ���ƶ���������Ҫ�ƶ���λ����Ϊֹ��
                                    
	%calculate gaussian response for all positions
	k = exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(x)));%��˹�ˣ�������Eq.16
%% ��ע�⵽k�Ǿ�����ʽ��������Ϊ�ڼ���Ĺ���������xû����������������xfҲ�Ǿ���                                       
end
                    %max����������ȷ��ֵ�������Ǹ�������Ϊ���������(x-y)^2������Ϊ��������Ϊ�˷�ֹ����ϵͳ���⣬����maxʹϵͳ���ȶ�
                    %������û�н���ΪʲôҪ����numel(x),����Ҫ������һ����numel(x)=numel(y).
                    %�پ��ǽ�������Ϊʲô����numel(x):ͨ������numel(x)�Ͳ�����������µ�mesh(k)����Ͳ鿴k��ֵ����������������numel(x),��ô�������ĵ���
                    %������λ�õ�k_i������������0��k�����ֱֵ��ȡ��0����͵����˺�k����Ϊ�������ý�ȡ��Ŀ�꣬�����������ΪĿ���ͼ�Ҳ����ˣ�ͨ�׽���
                    %���Ǹ������ĵ�ƫһ�㶼�����ԣ�³����̫�������numel(x)�󣬻Ὣ���е�ֵ����һ����С�ķ�Χ�ڣ�kֵ�����ƽ����³���Ը��á�
                    %ʹ��dog1���ݼ����в�����֤�˷��������Կ�������dog1��dog�ƶ�����ʱ�������Ը��ٵ�����΢�е�ƫ�ƾ͸��ٶ�ʧ�ˡ�
                    
