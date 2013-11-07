% Testing CAE penalty computations
clear all
eps = 1e-4;
JacCoeff = 1;
HessCoeff = 1;
HessBatchSize = 50;
HessNoise = .01;

ae = CAE('JacCoeff', JacCoeff, ...
         'HessCoeff', HessCoeff, ...
         'HessBatchSize', HessBatchSize, ...
         'HessNoise', HessNoise);
      
ae.encodeLayer = LogisticHiddenLayer(2, 2);
ae.decodeLayer = LogisticOutputLayer(2, 1);
ae.gather();

ae.encodeLayer.params{1} = [1 2; 3 4];
x = [1; .5];

% z = [2; 5], y = [.8808; .9933]
% Dy = 