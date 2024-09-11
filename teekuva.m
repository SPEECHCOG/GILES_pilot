function teekuva(filename, style, directory)
% Saves picture with filename to the default directory
%   Inputs:
%       filename    name of the picture file
%       style       eps_bw: black&white .eps,
%                   eps or eps_col: colored .eps,
%                   jpeg or jpg: .jpeg,
%                   png: .png,
%                   svg: .svg
%       directory   save to other than default directory

if nargin < 3
    directory = [];
end
if nargin < 2
    style = 'eps';
end

if(isempty(directory))
    filename1 = filename;
    filename2 = [filename '.fig'];
else
    filename1 = [directory '/' filename];
    filename2 = [directory '/' filename '.fig'];
end


set(gcf,'PaperPositionMode','auto')

if(strcmp(style, 'eps_bw'))
    print(filename1,'-deps','-loose');
elseif(strcmp(style, 'eps') || strcmp(style, 'eps_col'))
    print(filename1,'-depsc','-loose');
elseif(strcmp(style, 'jpg') || strcmp(style, 'jpeg'))
    print(filename1,'-djpeg90','-loose');
elseif(strcmp(style, 'png'))
    print(filename1,'-dpng','-loose');
elseif(strcmp(style, 'svg'))
    print(filename1,'-dsvg','-loose');
end

saveas(gcf, filename2, 'fig')
