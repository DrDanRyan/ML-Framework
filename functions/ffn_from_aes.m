function ffn = ffn_from_aes(aes, varargin)
ffn = FeedForwardNet(varargin{:});
nAE = length(aes);
ffn.hiddenLayers = cell(1, 2*nAE - 1);

for i = 1:nAE
   ffn.hiddenLayers{i} = aes{i}.encodeLayer.copy();
end

% Copy the hiddenLayer portion of the decodeLayer (assumes ComboOutputLayer type)
for i = 1:nAE-1
   ffn.hiddenLayers{nAE+i} = aes{nAE+1-i}.decodeLayer.hiddenLayer.copy();
end

ffn.outputLayer = aes{1}.decodeLayer.copy();
end

