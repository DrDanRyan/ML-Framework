function ffn = ffn_from_aes(aes, inputDropout, hiddenDropout)
% Stacks the encodeLayer from multiple AutoEncoders together to form a
% FeedForwardNet.
%
% aes = cell array of AutoEncoders
% 
% inputDropout = the amount of inputDropout that the FFN will use
% 
% hiddenDropout = the amount of hiddenDropout that the FFN will use

ffn = FeedForwardNet('inputDropout', inputDropout, ...
                     'hiddenDropout', hiddenDropout);
nAE = length(aes);
ffn.hiddenLayers = cell(1, 2*nAE - 1);
ffn.hiddenLayers{1} = aes{1}.encodeLayer.copy();
ffn.hiddenLayers{1}.params{1} = ffn.hiddenLayers{1}.params{1}/(1-inputDropout);

for i = 2:nAE
   ffn.hiddenLayers{i} = aes{i}.encodeLayer.copy();
   ffn.hiddenLayers{i}.params{1} = ...
      ffn.hiddenLayers{i}.params{1}/(1-hiddenDropout);
end

% Copy the hiddenLayer of the decodeLayer (assumes ComboOutputLayer type)
for i = 1:nAE-1
   ffn.hiddenLayers{nAE+i} = aes{nAE+1-i}.decodeLayer.hiddenLayer.copy();
   ffn.hiddenLayers{nAE+i}.params{1} = ...
      ffn.hiddenLayers{nAE+i}.params{1}/(1-hiddenDropout);
end

ffn.outputLayer = aes{1}.decodeLayer.copy();
ffn.outputLayer.hiddenLayer.params{1} = ...
   ffn.outputLayer.hiddenLayer.params{1}/(1-hiddenDropout);
end

