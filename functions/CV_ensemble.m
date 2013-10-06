function [outputs, testLosses] = CV_ensemble(inputs, targets, nFolds, models, ...
                                        trainer, sampler, preprocessors)
                                          
if nargin < 7
   preprocessors = [];
end

if isempty(targets)
   targets = inputs;
end

[outputSize, dataSize] = size(targets);
ensembleSize = length(models);
outputs = zeros(ensembleSize*outputSize, dataSize);
hold_outs = cross_validate_partition(dataSize, nFolds);

for foldIdx = 1:nFolds
   testIdx = hold_outs{foldIdx};
   trainSplit = setdiff(1:dataSize, testIdx);
   
   for i = 1:ensembleSize
      
      fprintf('\n\n Fold %d Model %d training started.\n', foldIdx, i);
      % Set model
      trainer.model = models{i};
      
      % Sample from training data and apply preprocessor to set dataManager
      trainIdx = sampler.sample(trainSplit, targets(:,trainSplit));
      validIdx = setdiff(trainSplit, trainIdx);
      if isempty(preprocessors)
         trainer.dataManager.trainingInputs = inputs(:, trainIdx);
         trainer.dataManager.validationInputs = inputs(:, validIdx);
      else
         trainer.dataManager.trainingInputs = preprocessors{i}.transform(inputs(:, trainIdx));
         trainer.dataManager.validationInputs = preprocessors{i}.transform(inputs(:, validIdx));
      end
      trainer.dataManager.trainingTargets = targets(:, trainIdx);
      trainer.dataManager.validationTargets = targets(:, validIdx);
      
      % Reset trainer and train model
      trainer.reset();
      trainer.train();
      
      % Make prediction for validation set
      startIdx = (i-1)*outputSize + 1;
      stopIdx = i*outputSize;
      
      if isempty(preprocessors)
         outputs(startIdx:stopIdx, testIdx) = gather(trainer.model.output(inputs(:,testIdx)));
      else
         outputs(startIdx:stopIdx, testIdx) = ...
            gather(trainer.model.output(preprocessors{i}.transform(inputs(:,testIdx))));
      end
   end
end

testLosses = zeros(ensembleSize, 1);
for i = 1:ensembleSize
   startIdx = (i-1)*outputSize + 1;
   stopIdx = i*outputSize;
   testLosses(i) = models{i}.compute_loss(outputs(startIdx:stopIdx, :), targets);
end

end

