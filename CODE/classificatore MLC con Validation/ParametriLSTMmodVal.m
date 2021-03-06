%% uso di validation set
numHiddenUnits = 100;
maxEpochs = 36;
miniBatchSize = 15;

layers = [ ...
    sequenceInputLayer(nVariables)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    reluLayer
    dropoutLayer
    fullyConnectedLayer(nLabels)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...,
    'Plots','training-progress', ...
    'ValidationData',{XVal,YVal});