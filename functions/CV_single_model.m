function [outputs, testLoss] = CV_single_model(inputs, targets, nFolds, train_model_function)

if isempty(targets) % no targets => AutoEncoder training - use inputs as targets
   targets = inputs;
end

[outputSize, dataSize] = size(targets);
hold_outs = CV_partition(dataSize, nFolds);
gpuState = GPUState();
outputs = gpuState.zeros(outputSize, dataSize, 'single');

for i = 1:nFolds
   testSplit = hold_outs{i};
   
   trainSplit = setdiff(1:dataSize, testSplit);
   trainIdx = sampler.sample(trainSplit, targets(:, trainSplit));
   validIdx = setdiff(trainSplit, trainIdx);
   model = train_model_function(inputs(:,trainIdx), targets(:,trainIdx),...
                                 inputs(:,validIdx), targets(:,validIdx));
   outputs(testSplit) = model.output(inputs(:, testSplit));
end

testLoss = model.compute_loss(outputs, targets);

end

