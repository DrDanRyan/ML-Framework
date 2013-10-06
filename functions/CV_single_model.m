function [outputs, testLoss] = CV_single_model(inputs, targets, nFolds, trainer, sampler)

if isempty(targets) % no targets => unsupervised AutoEncoder training
   targets = inputs;
end

[outputSize, dataSize] = size(targets);

hold_outs = CV_partition(dataSize, nFolds);
outputs = gpuArray.zeros(outputSize, dataSize, 'single');

for i = 1:nFolds
   testSplit = hold_outs{i};
   
   trainSplit = setdiff(1:dataSize, testSplit);
   trainIdx = sampler.sample(trainSplit, targets(:, trainSplit));
   validIdx = setdiff(trainSplit, trainIdx);
   trainer.dataManager.trainingInputs = inputs(:, trainIdx);
   trainer.dataManager.trainingTargets = targets(:, trainIdx);
   trainer.dataManager.validationInputs = inputs(:, validIdx);
   trainer.dataManager.validationTargets = targets(:, validIdx);
   trainer.reset();
   trainer.train();
   outputs(testSplit) = trainer.model.output(inputs(:, testSplit));
end

testLoss = trainer.model.compute_loss(outputs, targets);

end

