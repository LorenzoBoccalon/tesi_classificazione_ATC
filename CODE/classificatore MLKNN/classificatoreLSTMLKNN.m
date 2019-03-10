%% utilizzo del pacchetto MLkNN con LSTM
%viene utilizzato come classificatore il pacchetto MLkNN al posto di MLC

clear variables
warning off

%carico i dati
load ATC_42_3883.mat

nVariables = size(atc_fea,1);   %numero di features
nInstances = size(atc_fea,2);   %numero di istances
nLabels = size(atcClass, 1);    %numero di labels

Num = 11;                       %numero di neighbor 
Smooth = 1;                     %Laplace smoothing

ParametriLSTM;                  %carica parametri LSTM

SCORE = [];
lab = [];
sample_dim = floor(nInstances / 10);
index = 0;


for j = 1 : sample_dim + 1 : nInstances
    clear Outputs RisultatoO
    close all force
    index = index + 1;
    
    %% LSTM
    TR=atc_fea';
    YTrain=atcClass;
    %per tutti i pattern con più label inserisco nel training più
    %volte, considerando label diverse
    YTrain(:,j:min([j+sample_dim nInstances]))=[];
    TR(j:min([j+sample_dim nInstances]),:)=[];
    
    %duplico pattern per ogni sua classe
    clear label
    t=1;
    NTR=TR;
    for i=1:size(YTrain,2)
        %quante label ha il dato pattern?
        classi=find(YTrain(:,i)==1);
        if length(classi)>1
            for cl=1:length(classi)
                NTR(t,:)=TR(i,:);
                label(t)=classi(cl);
                t=t+1;
            end
        else
            NTR(t,:)=TR(i,:);
            label(t)=classi;
            t=t+1;
        end
    end
    TR=NTR;
    clear NTR
    YTrain=label;
    
    %TR tabella con 42 features, numero di istances superiore a 3883, non
    %ci sono sovrapposizioni
    
    
    %formatto i dati del training set in maniera compatibile per LSTM
    clear XTrain
    for i=1:size(TR,1)
        XTrain{i}=TR(i,:)';
    end
    %addestro la rete
    net = trainNetwork(XTrain,categorical(YTrain)',layers,options);
    %net è la rete addestrata
    layer = 'softmaxLayer';
    
    %formatto i dati del test set in maniera compatibile per LSTM
    TE=atc_fea(:,j:min([j+sample_dim nInstances]))';
    clear XTest
    for i=1:size(TE,1)
        XTest{i}=TE(i,:)';
    end
    
    
    %uso il layer softmaxLayer per rappresentare ogni pattern
    TR=atc_fea';
    TR(j:min([j+sample_dim nInstances]),:)=[];
    clear XTrain
    for i=1:size(TR,1)
        XTrain{i}=TR(i,:)';
    end
    
    %score LSTM
    testFeatures = predict(net,XTest, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');
    trainingFeatures = predict(net,XTrain, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');
    
    %elimino una feature se ha sempre lo stesso valore per tutti i
    %training pattern
    controllo=[];
    
    controllo(size(trainingFeatures,2))=0;
    for i=1:size(trainingFeatures,2)
        if length(unique(trainingFeatures(:,i)))==1
            controllo(i)=1;
        end
    end
    trainingFeatures(:, controllo>0)=[];
    testFeatures(:, controllo>0)=[];
        
    %% MLKNN
    YTest=atcClass;
    YTest(:,j:min([(j+sample_dim) nInstances]))=[];
    
    clear train_data
    clear train_target
    clear test_data
    clear test_target
    
    %features
    train_data = trainingFeatures;
    test_data = testFeatures;
    %label
    train_target = YTest;
    test_target = atcClass(:,j:min([(j+sample_dim) nInstances]));
    
    %training
    [Prior,PriorN,Cond,CondN] = MLKNN_train(train_data,train_target,Num,Smooth);
    %testing   
    [HammingLoss,RankingLoss,OneError,CoverageMLKNN,Average_Precision,Outputs,Pre_Labels] = ...
        MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
        
    SCORE=[SCORE; Pre_Labels'];
    lab=[lab atcClass(:,j:min([j+sample_dim nInstances]))];
        
end
%% performance values
[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = multi_labe_metrics(SCORE',lab);

save('classMLKNN.mat');

