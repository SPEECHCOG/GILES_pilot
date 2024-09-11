DATA_DIR = 'results/';
LEXTEST_DATA_DIR = 'lextest_results/';
abx_result_path = [DATA_DIR 'abxtest_all_results.mat'];



set(0,'defaulttextfontsize',18);
set(0,'defaultaxesfontsize',18);

include_CHILDES = 1;

load abx_result_path RCL EPC models

cds = ~cellfun(@isempty,strfind(models,'.cds.'));
ljs = ~cellfun(@isempty,strfind(models,'ljs.'));
exp2 = ~cellfun(@isempty,strfind(models,'exp2.'));
giles = ~cellfun(@isempty,strfind(models,'.giles.'));
childes = ~cellfun(@isempty,strfind(models,'.childes'));
both = ~cellfun(@isempty,strfind(models,'exp2-ljs'));
books = ~cellfun(@isempty,strfind(models,'librispeech'));

ljspeech = ~cellfun(@isempty,strfind(models,'ljspeech'));
libriorig = ~cellfun(@isempty,strfind(models,'orig.read.librispeech'));


to_remove = both | exp2 | ljspeech | libriorig;

cds(to_remove) = [];
ljs(to_remove) = [];
giles(to_remove) = [];
childes(to_remove) = [];
books(to_remove) = [];
both(to_remove) = [];

RCL(to_remove) = [];
EPC(to_remove) = [];
models(to_remove) = [];

random_baseline_abx = 0.3682;
random_baseline_lextest = 0.0431;

for k = 1:length(EPC)
    EPC{k}(1) = 0.5;
    EPC{k} = [0.15,EPC{k}];
    RCL{k} = [random_baseline_abx,RCL{k}];
    
    EPC{k} = EPC{k}(1:27);
    RCL{k} = RCL{k}(1:27);
end



RCL_CDS = nanmean(cell2mat(RCL(cds)));
RCL_NEUT = nanmean(cell2mat(RCL(~cds)));
RCL_READ = nanmean(cell2mat(RCL(books)));
RCL_GILES = nanmean(cell2mat(RCL(giles)));




figure(1);clf;
subplot(2,2,1);

plot(EPC{1},RCL_CDS.*100,'LineWidth',3);hold on;plot(EPC{1},RCL_NEUT.*100,'LineWidth',3)
legend({'CDS','Neutral'});
ylabel('ABX error (%)');
xlabel('epoch');
grid;
set(gca,'Xscale','log')
set(gca,'Yscale','log');
set(gca,'XTick',[0.15 0.5 2 10 50 100]);
set(gca,'XTickLabel',{'0','1','2','10','50','100'});
%ylim([0.13 0.35])
ylim([13 35])
title('speaking style');
xlim([0.1 50])
subplot(2,2,2);

plot(EPC{1},RCL_GILES.*100,'LineWidth',3);hold on;plot(EPC{1},RCL_READ.*100,'LineWidth',3)
legend({'generated CDS','Books'});
ylabel('ABX error (%)');
xlabel('epoch');
grid;
set(gca,'Xscale','log');
set(gca,'Yscale','log');
set(gca,'XTick',[0.15 0.5 2 10 50 100]);
set(gca,'XTickLabel',{'0','1','2','10','50','100'});
%ylim([0.13 0.35])
ylim([13 35])
title('speech content');

if(include_CHILDES)
    RR = nanmean(cell2mat(RCL(childes)));
    plot(EPC{1},RR.*100,'LineWidth',3,'LineStyle','--','DisplayName','original CHILDES','Color','black');
end

xlim([0.1 50])


%% Lextest part

RCL = cell(length(models),1);
EPC = cell(length(models),1);
for moditer = 1:length(models)    
        model_id = models{moditer};
        savedir = [LEXTEST_DATA_DIR model_id '/CDI_lextest/'];
        if(exist([savedir 'LEXresults_rnn_' model_id '.csv']))
            tmp = importdata([savedir 'LEXresults_rnn_' model_id '.csv']);
            RCL{moditer} = tmp(:,2)';
            EPC{moditer} = tmp(:,1)';
        end
end

cds = ~cellfun(@isempty,strfind(models,'.cds.'));
ljs = ~cellfun(@isempty,strfind(models,'ljs.'));
exp2 = ~cellfun(@isempty,strfind(models,'exp2.'));
giles = ~cellfun(@isempty,strfind(models,'.giles.'));
childes = ~cellfun(@isempty,strfind(models,'.childes'));
both = ~cellfun(@isempty,strfind(models,'exp2-ljs'));
books = ~cellfun(@isempty,strfind(models,'librispeech'));

ljspeech = ~cellfun(@isempty,strfind(models,'ljspeech'));
libriorig = ~cellfun(@isempty,strfind(models,'orig.read.librispeech'));


to_remove = both | exp2 | ljspeech | libriorig;

cds(to_remove) = [];
ljs(to_remove) = [];
giles(to_remove) = [];
childes(to_remove) = [];
books(to_remove) = [];
both(to_remove) = [];

RCL(to_remove) = [];
EPC(to_remove) = [];
models(to_remove) = [];

for k = 1:length(EPC)
    EPC{k}(1) = 0.5;
    EPC{k} = [0.15,EPC{k}];
    RCL{k} = [random_baseline_lextest,RCL{k}];
    
    EPC{k} = EPC{k}(1:27);
    RCL{k} = RCL{k}(1:27);
end


RCL_CDS = nanmean(cell2mat(RCL(cds)));
RCL_NEUT = nanmean(cell2mat(RCL(~cds)));
RCL_READ = nanmean(cell2mat(RCL(books)));
RCL_GILES = nanmean(cell2mat(RCL(giles)));



figure(1);
subplot(2,2,3);
plot(EPC{1},RCL_CDS.*100,'LineWidth',3);hold on;plot(EPC{1},RCL_NEUT.*100,'LineWidth',3)
legend({'CDS','Neutral'},'Location','SouthEast');
ylabel('Lextest score (%)');
grid;
set(gca,'Xscale','log')
set(gca,'Yscale','linear');
set(gca,'XTick',[0.15 0.5 2 10 50 100]);
set(gca,'XTickLabel',{'0','1','2','10','50','100'});
xlabel('epoch');
%ylim([0.1 0.25])
ylim([10 25])
xlim([0.1 100]);
xlim([0.1 50])
subplot(2,2,4);
plot(EPC{1},RCL_GILES.*100,'LineWidth',3);hold on;plot(EPC{1},RCL_READ.*100,'LineWidth',3)
legend({'generated CDS','Books'},'Location','SouthEast');
ylabel('Lextest score (%)');
grid;
set(gca,'Xscale','log')
set(gca,'Yscale','linear');
set(gca,'XTick',[0.15 0.5 2 10 50 100]);
set(gca,'XTickLabel',{'0','1','2','10','50','100'});
%ylim([0.1 0.25])
ylim([10 25])
xlim([0.1 100]);
xlabel('epoch');
xlim([0.1 50])

if(include_CHILDES)
    RR = nanmean(cell2mat(RCL(childes)));
    plot(EPC{1},RR.*100,'LineWidth',3,'LineStyle','--','DisplayName','original CHILDES','Color','black');
end


teekuva('CDI_lextest_and_ABXtest_GILES_all_epochs', 'eps')
teekuva('CDI_lextest_and_ABXtest_GILES_all_epochs', 'jpg')
