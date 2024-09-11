% violin plots: https://github.com/bastibe/Violinplot-Matlab

set(0,'defaultaxesfontsize',18);
set(0,'defaulttextfontsize',18);
close all;

% Finals
input_path = 'input/v2.2024';
fig_path = 'output/figs';
modelname = 'GILES_CHILDES_agecond_final_1word_5layers_512emb_512ff_8head_8000voc_5do_500k_100t_';
refname = 'childes_age_0';
if ~exist(fig_path, 'dir')
    mkdir(fig_path);
end

drawlinefits = 1;
separateplots = 0;
jointplots = 0;
newwordplots = 1;
valid_ages = [6,9,12,15,18,21,24,36,48];
valid_ages = [6,9,12,15,18,21,24,36,48];

disp('Loading data...')

a = dir(strcat(input_path, '/', modelname, '*.features'));
b = dir(strcat(input_path, '/', refname, '*.features'));

% Parse ages of references data
b_age = zeros(length(b),1);
for j = 1:length(b)
    b_age(j) = str2num(b(j).name(end-10:end-9));
end

fnam = [input_path '/' a(1).name];
D = importdata(fnam);

DATA = zeros(size(D.data,1),size(D.data,2),length(a),2);
age = zeros(length(a),1);


for k = 1:length(a)
    fnam = [input_path '/' a(k).name];
    % disp(fnam)

    D = importdata(fnam);
    DATA(:,:,k,1) = D.data;

    % Age of current model data
    age(k) = str2num(a(k).name(end-10:end-9));

    % Find the corresponding reference data based on age
    i = find(b_age == age(k));
    fnam2 = [input_path '/' b(i).name];
    Dref = importdata(fnam2);

    disp(sprintf([fnam(end-20:end) '\t' fnam2(end-20:end)]))
    
    % Find a mapping between feature columns of target and reference data
    target_column = zeros(length(D.colheaders),1);
    for j = 1:length(D.colheaders)
        target_feat = D.colheaders{j};
        tmp_i = find(strcmp(Dref.colheaders,target_feat));
        if(~isempty(tmp_i))
            target_column(j) = tmp_i;
        else
            [~, file_name, ~] = fileparts(fnam2);
            throw(MException('Value:Error', sprintf('No reference data for measure "%s" in CHILDES (%s)', target_feat, file_name)))
        end
    end
    DATA(:,:,k,2) = Dref.data(1:size(D.data,1),target_column);
end


% Remove age bins that are not of interest
toremove = [];
for k = 1:length(age)
    if(~sum(valid_ages == age(k)))
        toremove = [toremove;k];
    end
end
DATA(:,:,toremove,:) = [];
age(toremove) = [];
N_bins = length(age);

headers = D.colheaders;
featnames = cell(size(headers));
for k = 1:length(headers)
    s = headers{k};
    s = strrep(s,'_','\_');
    featnames{k} = s;
    featnames{k} = headers{k};
end

% Plot separate feature metrics
disp('Plotting metric data...')
if(separateplots)
    r = zeros(length(featnames),3);
    rp = zeros(length(featnames),3);
    pfit = zeros(length(featnames),2,2);
    for feat = 1:length(featnames)

        d1 = squeeze(DATA(:,feat,:,1));
        d2 = squeeze(DATA(:,feat,:,2));

        sig = zeros(size(d1,2),1);
        p = zeros(size(d1,2),1);
        tstat = cell(size(d1,2),1);
        dprime = zeros(size(d1,2),1);

        d_to_plot = zeros(size(d1,1),size(d1,2)*2);
        x = 1;
        for k = 1:size(d1,2)
            d_to_plot(:,x) = d1(:,k);
            d_to_plot(:,x+1) = d2(:,k);

            [sig(k),p(k),~,tstat{k}] = ttest(d1(:,k),d2(:,k));
            dprime(k) = ES_from_ttest(d1(:,k),d2(:,k));

            x = x+2;
        end

        fh = figure(2);clf;
        fh.Position = [2651 518 1223 506];
        h = violinplot(d_to_plot);

        for k = 1:2:size(DATA,3)*2-1
            h(k).ViolinColor = {[0 0.4470 0.7410]};
            h(k+1).ViolinColor = {[0.8500 0.3250 0.0980]};
            h(k).ScatterPlot.Visible = 0;
            h(k+1).ScatterPlot.Visible = 0;
            h(k).ViolinAlpha = {[0.45]};
            h(k+1).ViolinAlpha = {[0.45]};


        end
        set(gca,'XTick',[1.5:2:2*N_bins]);
        xt = xticks;
        set(gca,'XTickLabel',age);
        xlim([0 max(xt)+1.5]);
        xlabel('age (months)');
        ylabel('value');
        title(sprintf(featnames{feat}),'Interpreter','none');

        m1 = mean(d1);
        corrtype = 'Pearson';

        [r(feat,1),rp(feat,1)] = corr(age,mean(d1)','type',corrtype);
        [r(feat,2),rp(feat,2)] = corr(age,mean(d2)','type',corrtype);
        [r(feat,3),rp(feat,3)] = corr(mean(d1)',mean(d2)','type',corrtype);

        lh = legend({'','model','','','','','','','','CHILDES'});
        drawnow;
        teekuva(sprintf('%sGILES_CogSci_comparison_%s',fig_path, featnames{feat}), 'jpeg');
    end
end

% Plot all feature metrics in one figure
if (jointplots)
    % Features of interest and a corresponding figure name
    figure_name = 'GILES_comparison_all';
    f_interest = {
        'sent_length', 'utterance length';
        'ttr', 'TTR';
        'lm_perplexity', 'LM perplexity';
        'voc_rank_diff', 'lexical divergence';
        'complex_sent_rate', 'complex sent. rate';
        'synt_tree_height', 'synt. tree height';
        'mean_dep_per_token', 'dep. per word';
        'mean_dep_distance', 'dep. dist.';
        'pos_rate_NOUN', 'rate NOUN';
        'pos_rate_VERB', 'rate VERB';
        'pos_rate_ADJ',  'rate ADJ';
        'pos_rate_ADV', 'rate ADV';
        'pos_ttr_NOUN', 'TTR NOUN';
        'pos_ttr_VERB', 'TTR VERB';
        'pos_ttr_ADJ',  'TTR ADJ';
        'pos_ttr_ADV', 'TTR ADV';
        'pos_rate_INTJ', 'rate INTJ';
        'pos_rate_PRON', 'rate PRON';
        'pos_rate_DET',  'rate DET';
        'pos_rate_CONJ', 'rate CONJ';
        'pos_ttr_INTJ', 'TTR INTJ';
        'pos_ttr_PRON', 'TTR PRON';
        'pos_ttr_DET',  'TTR DET';
        'pos_ttr_CONJ', 'TTR CONJ'
        };

    n_features = size(f_interest, 1);
    
    fgh = figure(5);
    clf;
    fgh.Position = [500 500 ceil(n_features/2)*300 ceil(n_features/2)*100+150];
    if strcmp(input_path, 'input/v1.2023')
        fgh.Position = [887 859 1313 478];
    elseif n_features > 10
        fgh.Position = [50 50 900 1500];
        fgh.Position = [50 50 900 1500];
    end
    
    if (n_features <= 10)
        ylabelHandles = gobjects(2, 1);
    else
        ylabelHandles = gobjects(ceil(n_features/4), 1);
    end
    for iter = 1:n_features
        if (n_features <= 10)
            subplot(2,ceil(n_features/2),iter);
        else
            subplot(ceil(n_features/4),4,iter);
        end
        feat = cellfind(headers,f_interest{iter,1}, 'exact');
    
        d1 = squeeze(DATA(:,feat,:,1));
        d2 = squeeze(DATA(:,feat,:,2));
    
        sig = zeros(size(d1,2),1);
        p = zeros(size(d1,2),1);
        tstat = cell(size(d1,2),1);
        dprime = zeros(size(d1,2),1);
    
        d_to_plot = zeros(size(d1,1),size(d1,2)*2);
        x = 1;
        for k = 1:size(d1,2)
            d_to_plot(:,x) = d1(:,k);
            d_to_plot(:,x+1) = d2(:,k);
            [sig(k),p(k),~,tstat{k}] = ttest(d1(:,k),d2(:,k));
            dprime(k) = ES_from_ttest(d1(:,k),d2(:,k));
            x = x+2;
        end
    
        h = violinplot(d_to_plot);
        for k = 1:2:size(DATA,3)*2-1
            h(k).ViolinColor = {[0 0.4470 0.7410]};
            h(k+1).ViolinColor = {[0.8500 0.3250 0.0980]};
            h(k).ScatterPlot.Visible = 0;
            h(k+1).ScatterPlot.Visible = 0;
            h(k).ViolinAlpha = {[0.75]};
            h(k+1).ViolinAlpha = {[0.75]};
        end
        set(gca,'XTick',[1.5:2:2*N_bins]);
            xt = xticks;
            set(gca,'XTickLabel',age);
            xlim([0 max(xt)+1.5]);
    
        if n_features<= 10 && iter > n_features/2
            xlabel('age (months)');
        elseif iter > floor(n_features/4) * 4 - 4
            xlabel('age (months)');
        end
    
        if iter == 1
            ylabelHandles(1) = ylabel('value');
        elseif n_features<= 10 && iter == ceil(1+n_features/2)
            ylabelHandles(2) = ylabel('value');
        elseif mod(iter, 4) - 1  == 0
            ylabelHandles(ceil(iter/4)) = ylabel('value');
        end
        title(sprintf(f_interest{iter,2}),'Interpreter','none');
    
        m1 = mean(d1);
        a = polyfit(xt,m1,2);
        if(drawlinefits)
            plot(xt,xt.^2*a(1)+xt.*a(2)+a(3),'--','Color','blue','LineWidth',2);
        end
    
        m2 = mean(d2);
        a2 = polyfit(xt,m2,2);
        if(drawlinefits)
            plot(xt,xt.^2*a2(1)+xt.*a2(2)+a2(3),'--','Color','red','LineWidth',2);
        end
    
        corrtype = 'Pearson';
        [r(feat,1),rp(feat,1)] = corr(age,mean(d1)','type',corrtype);
        [r(feat,2),rp(feat,2)] = corr(age,mean(d2)','type',corrtype);
        [r(feat,3),rp(feat,3)] = corr(mean(d1)',mean(d2)','type',corrtype);
        drawnow;
        
        if(iter == 1)
            lh = legend({'','model','','','','','','','','CHILDES'},'Location','NorthWest');
        end
    
    end
    positions = cell2mat(get(ylabelHandles, 'Position'));
    minX = min(positions(:, 1));
    for i = 1:length(ylabelHandles)
        positions(i, 1) = minX;
        set(ylabelHandles(i), 'Position', positions(i, :));
    end
    disp('Saving feature figures...')
    teekuva(figure_name, 'eps', fig_path);
    teekuva(figure_name, 'jpeg', fig_path);
    teekuva(figure_name, 'png', fig_path);
end


disp('Plotting new word figures...')
% Print new tokens plot
if (newwordplots)
    % Create overlap figs
    olap_path = strcat(input_path, '/../', 'overlaps_500k_1word.csv');
    D = importdata(olap_path);
    ages = D.data(:,1);
    uttlen = D.data(:,2);
    propuq = D.data(:,3);
    count = D.data(:,4);
    
    uq_ages = unique(ages);
    
    P = zeros(length(uq_ages),max(uttlen));
    C = zeros(length(uq_ages),max(uttlen));
    
    for k = 1:length(ages)
        i = find(uq_ages == ages(k)); 
        P(i,uttlen(k)) = propuq(k);
        C(i,uttlen(k)) = count(k);
    end
    
    maxlen = 14;
    P = P(:,1:maxlen);
    C = C(:,1:maxlen);
    
    % 
    olap_path = strcat(input_path, '/../', 'overlaps_CHILDES_1word_fixed.csv');
    D2 = importdata(olap_path);
    
    ages2 = D2.data(:,1);
    uttlen2 = D2.data(:,2);
    propuq2 = D2.data(:,3);
    count2 = D2.data(:,4);
    
    uq_ages2 = unique(ages2);
    
    P2 = zeros(length(uq_ages2),max(uttlen2));
    C2 = zeros(length(uq_ages2),max(uttlen2));
    
    for k = 1:length(ages2)
        i = find(uq_ages2 == ages2(k)); 
        P2(i,uttlen2(k)) = propuq2(k);
        C2(i,uttlen2(k)) = count2(k);
    end
    
    maxlen = 14;
    P2 = P2(:,1:maxlen);
    C2 = C2(:,1:maxlen);
    
    %plot(uq_ages,mean(P),'LineWidth',2);
    
    h = figure(7);
    clf;
    hold on;
    h.Position = [500 500 330 300];
    h.Position = [2801 561 556 286];
    h.Position = [1000 1000 556 400];
    
    xlabel('utterance length (words)');
    ylabel('proportion new (%)')
    grid;
    xlim([0.5 maxlen+0.5])
    ylim([0 105])

    %line([0.5,maxlen+0.5],[100,100],'LineStyle','--','LineWidth',2,'Color','red');
    
    hold on;
    plot(mean(P),'LineWidth',3);
    plot(nanmean(P2),'LineWidth',3);
    drawstds(h,(1:maxlen)-0.1,nanmean(P),nanstd(P),0.25,1.5,'black');
    drawstds(h,(1:maxlen)+0.1,nanmean(P2),nanstd(P2),0.25,1.5,'black');
    legend({'model','CHILDES'},'Location','SouthEast')
    
    teekuva('GILES_generatednew_wordforms', 'eps', fig_path);
    teekuva('GILES_generatednew_wordforms', 'jpeg', fig_path);
    teekuva('GILES_generatednew_wordforms', 'png', fig_path);
end

disp('Done')