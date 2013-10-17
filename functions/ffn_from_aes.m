function ffn = ffn_from_aes(aes, inputDropout, hiddenDropout)
ffn = FeedForwardNet('inputDropout', inputDropout, ...
                     'hiddenDropout', hiddenDropout);
nAE = length(aes);
ffn.hiddenLayers = cell(1, 2*nAE - 1);



for 2 = 1:nAE
   ffn.hiddenLayers{i} = aes{i}.encodeLayer.copy();
   ffn.hiddenLayers{i}.params{1} = ffn.hiddenLayers{i}.params{1}/(1-hiddenDropout);
end

% Copy the hiddenLayer portion of the decodeLayer (assumes ComboOutputLayer type)
for i = 1:nAE-1
   ffn.hiddenLayers{nAE+i} = aes{nAE+1-i}.decodeLayer.hiddenLayer.copy();
end

ffn.outputLayer = aes{1}.decodeLayer.copy();
end

