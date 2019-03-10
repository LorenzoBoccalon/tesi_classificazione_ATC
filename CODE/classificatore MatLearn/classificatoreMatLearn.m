%% classificatore matLearn senza LSTM
% Demonstrates use of independent logistic regression classifiers for each 
% candidate class for multilabel classification. Each combination labels is
% represented by a unique color in the output plot and as a unique integer
% in [1,2^N] where N is the number of classes
clear all
close all

%% Load data

load ATC_42_3883.mat

nVariables = size(atc_fea,1);  %numero di features
nInstances = size(atc_fea,2);  %numero di istances
nLabels = size(atcClass, 1);   %numero di labels

sampleDim = floor(nInstances / 10);
index = 0;
SCORE = [];
lab = atcClass;

%% 10-fold

for j = 1 : sampleDim + 1 : nInstances
    %% Preprocess Data
    
    index = index + 1;
    X = atc_fea';
    y = binary2LinearInd(atcClass');

    % Standardize features and add bias
    X = standardizeCols(X);
    X = [ones(size(X,1),1) X];

    % Split into training/test set 
    Xtest = X(j:min([j+sampleDim nInstances]),:);
    ytest = y(j:min([j+sampleDim nInstances]));
    Xtrain = X;
    Xtrain(j:min([j+sampleDim nInstances]),:) = [];
    ytrain = y;
    ytrain(j:min([j+sampleDim nInstances]),:) = [];

    %% usage of independent logistic regression with L2-regularization
    
    options = struct('nLabels',nLabels,'lambdaL2',1e-4);
    model = ml_multilabel_independent(Xtrain,ytrain,options);
    yhatTest = model.predict(model, Xtest);
    yhatTrain = model.predict(model, Xtrain);
    testError = sum(ytest~=yhatTest)/length(ytest);
    model.trainError = sum(ytrain~=yhatTrain)/length(ytrain);
    fprintf('cicle %i: Averaged misclassification test error with %s is: %.3f\n',...
            index, model.name, testError);
    SCORE = [SCORE linearInd2Binary(yhatTest,nLabels)'];
end

%% Performance values

[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = multi_labe_metrics(SCORE,lab);

save('classMatLearn.mat');

