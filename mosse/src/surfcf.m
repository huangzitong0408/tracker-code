
function surfcf(varargin)
% ���ƴ�����ֵ��ͼ������ͼ���൱��surf+contourf
 
    hold on
    % plot the surface
    surf(varargin{:});
    shading interp; %��ֵ��Ӱģʽ
 
    % plot filled contour and get handle to hggroup object
    [C,h] = contourf(varargin{:});
    c = get(h, 'Children');
 
    % set the Z-data for each patch object to lower limit of Z-axis
    zmin = min(zlim);
    for i = 1:length(c)
        set(c(i), 'zdata', zmin*ones(size(get(c(i), 'xdata'))));
    end
    % ȥ����ֵ��ͼ��������ֻ�������ɫ��ʾ
    for ii = 1:length(h)
        set(h(ii), 'LineStyle', 'none');
    end
 
    view(-50, 30); %����3d�ӽ�
 
end