%% classificatore matLearn con LSTM
% Demonstrates use of independent logistic regression classifiers for each 
% candidate class for multilabel classification. Each combination labels is
% represented by a unique color in the output plot and as a unique integer
% in [1,2^N] where N is the number of classes
% 
% LSTM used to extract features and reduce number of variables for 
% logistic regression
%% Load and initialize data
clear all
close all

load ATC_42_3883.mat

nVariables = size(atc_fea,1);  %numero di features
nInstances = size(atc_fea,2);  %numero di istances
nLabels = size(atcClass, 1);   %numero di labels

sampleDim = floor(nInstances / 10);
index = 0;
SCORE = [];
lab = [];

%% parametri LSTM
numHiddenUnits = 100;
maxEpochs = 36;
miniBatchSize = 15;

layers = [ ...
    sequenceInputLayer(nVariables)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(nLabels)
    softmaxLayer
    classificationLayer];

LSTMoptions = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

%% 10-fold

for j = 1 : sampleDim + 1 : nInstances
    %indice del fold
    index = index + 1;
    
    %% estrazione feature LSTM %%
    
    clear Outputs RisultatoO
    close all force
    
    TR=atc_fea';
    YTrain=atcClass;
    %per tutti i pattern con più label inserisco nel training più
    %volte, considerando label diverse
    YTrain(:,j:min([j+sampleDim nInstances]))=[];
    TR(j:min([j+sampleDim nInstances]),:)=[];
    
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
    net = trainNetwork(XTrain,categorical(YTrain)',layers,LSTMoptions);
    %net è la rete addestrata
    layer = 'softmaxLayer';
    
    %formatto i dati del test set in maniera compatibile per LSTM
    TE=atc_fea(:,j:min([j+sampleDim nInstances]))';
    clear XTest
    for i=1:size(TE,1)
        XTest{i}=TE(i,:)';
    end
    
    
    %uso il layer softmaxLayer per rappresentare ogni pattern
    TR=atc_fea';
    TR(j:min([j+sampleDim nInstances]),:)=[];
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
        
    %% classificatore matLearn %%    
    % Preprocess Data
    y = binary2LinearInd(atcClass');
    
    clear Xtest
    clear Xtrain
    clear ytest
    clear ytrain
    
    % Standardize features and add bias
    % test set 
    Xtest = testFeatures;
    Xtest = standardizeCols(Xtest);
    Xtest = [ones(size(Xtest,1),1) Xtest]; 
    ytest = y(j:min([j+sampleDim nInstances]));
    
    % training set 
    Xtrain = trainingFeatures;
    Xtrain = standardizeCols(Xtrain);
    Xtrain = [ones(size(Xtrain,1),1) Xtrain];
    ytrain = y;
    ytrain(j:min([j+sampleDim nInstances]),:) = [];

    % usage of independent logistic regression with L2-regularization
    options = struct('nLabels',nLabels,'lambdaL2',1e-4);
    
    %training
    model = ml_multilabel_independent(Xtrain,ytrain,options);
    
    %testing
    yhatTest = model.predict(model, Xtest);
    yhatTrain = model.predict(model, Xtrain);
    testError = sum(ytest~=yhatTest)/length(ytest);
    model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
    fprintf('cicle %i: Averaged misclassification test error with %s is: %.3f\n',...
            index, model.name, testError);
        
    SCORE = [SCORE linearInd2Binary(yhatTest,nLabels)'];
    lab = [lab atcClass(:,j:min([j+sampleDim nInstances]))];
end
%% Performance values

[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = multi_labe_metrics(SCORE,lab);

save('classLSTMatLearn.mat');


