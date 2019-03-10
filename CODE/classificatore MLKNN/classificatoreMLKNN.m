%% utilizzo del pacchetto MLkNN senza LSTM
%viene utilizzato come classificatore il pacchetto MLkNN direttamente su ATC,
%senza usare LSTM o altro

clear variables
warning off

%carico i dati
load ATC_42_3883.mat

nVariables = size(atc_fea,1);   %numero di features
nInstances = size(atc_fea,2);   %numero di istances
nLabels = size(atcClass, 1);    %numero di labels

Num = 11;                       %numero di neighbor 
Smooth = 1;                     %Laplace smoothing

SCORE = [];
lab = [];
sample_dim = floor(nInstances / 10);
index = 0;

%% MLkNN
for j = 1 : sample_dim + 1 : nInstances
    index = index + 1;
    clear train_data
    clear train_target
    clear test_data
    clear test_target 
    
    %train_data
    train_data = atc_fea;
    train_data(:,j:min([(j+sample_dim) nInstances])) = [];
    train_data = train_data';
    %train_target
    train_target = atcClass;
    train_target(:,j:min([(j+sample_dim) nInstances])) = [];
    %test_data
    test_data = atc_fea(:,j:min([(j+sample_dim) nInstances]))';
    %test_target
    test_target = atcClass(:,j:min([(j+sample_dim) nInstances]));
    
    %training
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
    %testing
    [HammingLoss,RankingLoss,OneError,CoverageMLKNN,Average_Precision,Outputs,Pre_Labels] = ...
        MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
    
    SCORE = [SCORE; Pre_Labels'];
    lab = [lab atcClass(:,j:min([j+sample_dim nInstances]))];
end
%% performance values 

[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = multi_labe_metrics(SCORE',lab);
save('pureMLKNN.mat');
