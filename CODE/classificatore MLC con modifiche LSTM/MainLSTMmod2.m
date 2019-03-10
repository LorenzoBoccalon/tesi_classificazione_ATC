%% classificatore MLC con LSTM (modifica 2)
clear variables
warning off

%carico i dati
load ATC_42_3883.mat

nVariables = size(atc_fea,1);   %numero di features
nInstances = size(atc_fea,2);   %numero di istances
nLabels = size(atcClass, 1);    %numero di labels

ParametriMLC;                   %carica parametri MLC
ParametriLSTMmod2;              %carica parametri LSTM

lab = [];
SCORE = [];
sample_dim = floor(nInstances / 10);
index = 0;


for j = 1 : sample_dim + 1: nInstances
    index = index + 1;
    clear Outputs RisultatoO
    close all force
    
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
    
    %salva il chunk di previsione fatta da LSTM
    filename = {'C:\Users\loryt\Dropbox\TesiBoccalon\CODE\testFeatureCollection\testFeatureNo', num2str(index), '.mat'};
    filename = strcat(filename{1}, filename{2}, filename{3});
    save(filename, 'testFeatures');
        
    %% MLC
    clear RisultatoO
    YTest=atcClass;
    YTest(:,j:min([j+sample_dim nInstances]))=[];
    YTest=YTest';
    YTest(YTest==-1)=0;
    
    %training
    [model,train_time]=MLC_train(trainingFeatures,YTest,method);
    %testing
    [conf,test_time]=MLC_test(trainingFeatures,YTest,testFeatures,model,method);
        
    SCORE=[SCORE; conf];
    lab=[lab atcClass(:,j:min([j+sample_dim nInstances]))];
    
    save('SALVO.mat','SCORE','lab');%salva ad ogni step i risultati intermedi, per evitare di perderli se si blocca qualcosa
    
end
%% calcolo performance values
Presunte=zeros(size(SCORE));
Presunte(SCORE>0.5)=1;%presunte label decise dal classificatore
Presunte(SCORE<=0.5)=-1;
[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = multi_labe_metrics(Presunte',lab);

save('MainLSTMmod2.mat');
