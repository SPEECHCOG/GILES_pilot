RESULTS_DIR = 'results/';
DATA_DIR = 'abx_data/';
SERVER_DATA_DIR = 'server://large_data/giles/results/';
result_path = [DATA_DIR 'abxtest_all_results.mat'];

models=["exp2.neutral.librispeech.100" "exp2.cds.librispeech.100" "ljs.neutral.librispeech.100" "ljs.cds.librispeech.100" "exp2.neutral.giles.100" "exp2.cds.giles.100" "ljs.neutral.giles.100" "ljs.cds.giles.100" "orig.read.librispeech.100" "ljs.read.ljspeech.24" "orig.read.ljspeech.24" "exp2-ljs.neutral.giles.100" "ljs.neutral.childes.100" "ljs.cds.childes.100"];

RCL = cell(length(models),1);
EPC = cell(length(models),1);

redo_all = 1;

if(~redo_all)
    load result_path RCL EPC models
end

for moditer = 1:length(models)

    if(isempty(EPC{moditer}) || redo_all)
        model_id = models{moditer};

        tmpdir = [DATA_DIR model_id  '/ABX/'];

        if(exist(tmpdir,'dir'))
            rmdir(tmpdir,'s');
            mkdir(tmpdir)
        else
            mkdir(tmpdir)
        end


        [a,b] = system(['scp -r ' SERVER_DATA_DIR model_id '/ABX ' DATA_DIR model_id '/']);

        if(~contains(b,'No such file'))

            checkpoint_path = [DATA_DIR model_id '/ABX/'];
            savedir = [DATA_DIR model_id '/ABX/'];

            recall = [];
            epoch = [];
            tmp = dir(checkpoint_path);
            x = 1;
            for k = 1:length(tmp)
                if(find(strfind(tmp(k).name,'checkpoint')))
                    epoch(x) = str2num(tmp(k).name(strfind(tmp(k).name,'_')+1:end));
                    filename = [checkpoint_path tmp(k).name '/ABX_scores.json'];
                    jsonText = fileread(filename); % Read the file content
                    data = jsondecode(jsonText); % Parse the JSON content
                    recall(x) = data.within;
                    x = x+1;
                end
            end

            [epoch,i] = sort(epoch,'ascend');
            recall = recall(i);

            if(~exist(savedir,'dir'))
                mkdir(savedir)
            end
            csvwrite([savedir 'ABXresults_' model_id '.csv'],[epoch' recall']);

            RCL{moditer} = recall;
            EPC{moditer} = epoch;

        end
    end

end


save result_path RCL EPC models