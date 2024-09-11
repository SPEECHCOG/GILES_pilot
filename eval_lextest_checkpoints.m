HOME_DIR = '<USER HOME DIR>';
RESULTS_DIR = 'results/';
LEXTEST_DATA_DIR = 'lextest_results/';
DATA_DIR = 'abx_data/';
SERVER_DATA_DIR = 'server://large_data/giles/results/';
LEXTEST_MODEL_DIR = 'model_dir/CDI_lextest/CDI_synth';
DOWNLOAD_DIR = [HOME_DIR 'Downloads/'];
result_path = [RESULTS_DIR 'lextest_all_results.mat'];
result_rnn_path = [RESULTS_DIR 'lextest_all_results_RNN.mat'];

models=["exp2.neutral.librispeech.100" "exp2.cds.librispeech.100" "ljs.neutral.librispeech.100" "ljs.cds.librispeech.100" "exp2.neutral.giles.100" "exp2.cds.giles.100" "ljs.neutral.giles.100" "ljs.cds.giles.100" "orig.read.librispeech.100" "ljs.read.ljspeech.24" "orig.read.ljspeech.24" "exp2-ljs.neutral.giles.100" "ljs.neutral.childes.100" "ljs.cds.childes.100"];

RCL = cell(length(models),1);
EPC = cell(length(models),1);

redo_all = 1;
rnn_output = 1;


if(~redo_all)
    if(rnn_output)
        load result_rnn_path RCL EPC models
    else
        load result_path RCL EPC models
    end
end

for moditer = 1:length(models)
    if(isempty(EPC{moditer}) || redo_all)
        model_id = models{moditer};

        tmpdir = [DATA_DIR model_id  '/lextest_latents'];

        if(exist(tmpdir,'dir'))
            rmdir(tmpdir,'s');
            mkdir(tmpdir)
        else
            mkdir(tmpdir)
        end

        if(rnn_output)
            [a,b] = system(['scp -r ' SERVER_DATA_DIR model_id '/lextest_latents_false/lextest_latents.zip ' DATA_DIR model_id '/']);
        else
            [a,b] = system(['scp -r ' SERVER_DATA_DIR model_id '/lextest_latents/lextest_latents.zip ' DATA_DIR model_id '/']);
        end

        if(~contains(b,'No such file'))
            fprintf('Calculating CDI-lextest for %s\n',model_id);

            if(rnn_output)
                [c,d] = system(['unzip -qo ' DATA_DIR model_id  '/lextest_latents.zip -d ' DATA_DIR model_id  '/lextest_latents_false/']);
            else
                [c,d] = system(['unzip -qo ' DATA_DIR model_id  '/lextest_latents.zip -d ' DATA_DIR model_id  '/lextest_latents/']);
            end

            if(rnn_output)
                checkpoint_path = [DATA_DIR model_id '/lextest_latents_false/'];
            else
                checkpoint_path = [DATA_DIR model_id '/lextest_latents/'];
            end
            savedir = [LEXTEST_DATA_DIR model_id '/CDI_lextest/'];

            recall = [];
            epoch = [];
            tmp = dir(checkpoint_path);
            x = 1;
            for k = 1:length(tmp)
                if(find(strfind(tmp(k).name,'checkpoint')))
                    epoch(x) = str2num(tmp(k).name(strfind(tmp(k).name,'_')+1:end));
                    recall(x) = CDI_lextest(LEXTEST_MODEL_DIR,[checkpoint_path '/' tmp(k).name  '/avg/'],'single',1, DOWNLOAD_DIR);
                    x = x+1;
                end
            end

            [epoch,i] = sort(epoch,'ascend');
            recall = recall(i);

            if(~exist(savedir,'dir'))
                mkdir(savedir)
            end

            if(rnn_output)
                csvwrite([savedir 'LEXresults_rnn_' model_id '.csv'],[epoch' recall']);
            else
                csvwrite([savedir 'LEXresults_' model_id '.csv'],[epoch' recall']);
            end



            system(['scp -r ' savedir ' ' SERVER_DATA_DIR model_id '/']);

            RCL{moditer} = recall;
            EPC{moditer} = epoch;
        else
            fprintf('Skipping %s (no latents found)\n',model_id);
        end
    end

end


if(rnn_output)
    save result_rnn_path RCL EPC models
else
    save result_path RCL EPC models
end



