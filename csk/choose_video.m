function video_path = choose_video(base_path)
%CHOOSE_VIDEO
%   Allows the user to choose a video (sub-folder in the given path).
%   Returns the full path to the selected sub-folder.
%
%   Jo?o F. Henriques, 2012
%   http://www.isr.uc.pt/~henriques/

	%process path to make sure it's uniform
	
    if ispc() %ispc�����жϵ�ǰ�ĵ���ϵͳ�Ƿ���windowsϵͳ���Ƿ���1�����Ƿ���0 
        base_path = strrep(base_path, '\', '/');%strrep���Ҳ��滻���ַ���,��str�г��ֵ�����old���滻Ϊnew.newStr=strrep(str,old,new)
    end
    
    if base_path(end) ~= '/' %����base_path
        base_path(end+1) = '/'; 
    end
	
	%list all sub-folders
	contents = dir(base_path);%���ص�ǰ·���е������ļ��Լ��ļ�������ɵ��б�(���������'.'��'..')
	names = {};%�����洢�����ļ��е�name
	for k = 1:numel(contents)%numel����������Ԫ�ظ���
		name = contents(k).name;
		if isfolder([base_path name]) && ~strcmp(name, '.') && ~strcmp(name, '..')%MATLAB���齫isdir��Ϊisfolder; strcmpΪ�ַ����ȽϺ���
			names{end+1} = name;  %#ok
		end
	end
	
	%no sub-folders found
	
    if isempty(names)%�����ǰ·����û���ļ��� 
        video_path = []; 
        return; 
    end
	
	%choice GUI
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');%�б�ѡ��Ի��� Listdlg
	
	if isempty(choice)  %user cancelled
		video_path = [];
	else
		video_path = [base_path names{choice} '/'];
	end
	
end

