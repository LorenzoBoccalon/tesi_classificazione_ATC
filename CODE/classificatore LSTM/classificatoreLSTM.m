%% classificatoreLSTM 
% unisce le test features di LSTM create durante l'addestramento in MainLSTM 
% e tramite una soglia arbitraria decide le label, lo score del 
% classificatore viene salvato in PresunteLSTM.

%clear all
clear variables
warning off

numTF = 10;
soglia = 0.1;
TF = [];
%TF = zeros(1,14); %prima riga di 0 è intutile
%aggregare le testfeatures (TF)(10)
for i=1:numTF
    filename = {'C:\Users\loryt\Dropbox\TesiBoccalon\CODE\testFeatureCollection\testFeatureNo', num2str(i), '.mat'};
    filename = strcat(filename{1}, filename{2}, filename{3});
    load(filename);
    %TF = cat(1, TF, testFeatures);
    TF = [TF; testFeatures];
end
%elimino prima riga
%TF(1,:) = [];
%TF ora contiene tutte le previsioni di LSTM

PresunteLSTM=zeros(size(TF));
PresunteLSTM(TF>soglia)=1;%presunte label decise dal classificatore LSTM
PresunteLSTM(TF<=soglia)=-1;

load SALVO.mat

[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = multi_labe_metrics(PresunteLSTM',lab);

filename = "C:\Users\loryt\Dropbox\TesiBoccalon\CODE\classificatore LSTM\classLSTM.mat";
save(filename, 'PresunteLSTM');