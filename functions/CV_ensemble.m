function [outputs, testLosses] = CV_ensemble(inputs, targets, nFolds, ensembleSize, ...
                                             train_model_function, preprocessors)
                                          
if nargin < 6
   preprocessors = [];
end

if isempty(targets)
   targets = inputs;
end

[outputSize, dataSize] = size(targets);
gpuState = GPUState();
outputs = gpuState.zeros(ensembleSize*outputSize, dataSize);
testLosses = gpuState.zeros(ensembleSize, 1);
for i = 1:ensembleSize
   startIdx = (i-1)*outputSize + 1;
   stopIdx = i*outputSize;
   if ~isempty(preprocessors)
      transInputs = preprocessors{i}.transform(inputs);
   else
      transInputs = inputs;
   end
   
   fprintf('Cross-validating model %d ...\n', i);
   [outputs(startIdx:stopIdx, :), testLosses(i)] = CV_single_model(transInputs, targets, ...
                                                               nFolds, train_model_function);
end

end

