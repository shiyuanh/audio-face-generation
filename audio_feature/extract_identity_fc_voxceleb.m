clear all
run /home/shiyuan/matconvnet/matlab/vl_setupnn.m 	% Path to MatConvNet Vers 23
vl_contrib setup VGGVox

opts.modelPath = '' ;
opts.gpu = 0;
opts.dataDir = '/home/linxd/gan/data/wav';

% Load or download the VGGVox model for Identification
modelName = 'vggvox_ident_net.mat' ;
paths = {opts.modelPath, ...
    modelName, ...
    fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

if isempty(ok)
    fprintf('Downloading the VGGVox model for Identification ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
    mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
    url = sprintf('%s/~vgg/data/voxceleb/models/%s', base, modelName) ;
    urlwrite(url, opts.modelPath) ;
else
    opts.modelPath = paths{ok} ;
end

tmp = load(opts.modelPath); net = tmp.net ;

buckets.pool 	= [2 5 8 11 14 17 20 23 27 30];
buckets.width 	= [100 200 300 400 500 600 700 800 900 1000];

% Evaluate network either on CPU or GPU and set up network to be in test
% mode
if ~isempty(opts.gpu),net = vl_simplenn_move(net,'gpu'); end
net.mode = 'test';
net.conserveMemory = false;


fid = fopen('/home/shiyuan/BEGAN/voxceleb/track_info/speakers_split.txt');
C = textscan(fid, '%s %s %s %s');
field = {'name','id','url','split'};
af_table = cell2struct(C, field, 2);
    
    

%% Load and prepare data

audiofc = [];
for i=1:size(af_table.name,1)
    name = af_table.name{i};
    id = af_table.id{i}
    url = af_table.url{i};
    if str2num(id(end-3:end)) >= 270 & str2num(id(end-3:end)) <310
        continue
    end
    wavfile = fullfile(opts.dataDir, id, url, '00001.wav'); 
    inp = test_getinput(wavfile,net.meta,buckets);
    s1 = size(inp, 2);
    p1 = buckets.pool(s1==buckets.width);
    net.layers{16}.pool=[1 p1];
    res = vl_simplenn(net,inp);
    fc = gather(squeeze(res(18).x(:,:,:,:)));
    audiofc = [audiofc;fc'];
    size(audiofc)
    fprintf('i=%d, current sp: %s\n', i, name);
end


dlmwrite('audiofc_1024_2.txt', audiofc, 'delimiter',' ');

%for i=1:size(names,1)
%	if names(i).name(1) ~= '.'
%		tracks = dir([facefolder '/' names(i).name '/1.6/']);
%		for t=1:size(tracks,1)
%			if tracks(t).name(1) ~= '.'
%                name_id=find(strcmp(speakers_name,names(i).name));
%                if name_id >= 270 & name_id <= 309
%                    continue
%                end
%                export_audio(names(i).name, ['id' int2str(name_id+10000)],  tracks(t).name, opt, net, %aud_id, f);
                return
%			end
%		end
%			
%	end
	

%end


