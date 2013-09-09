function outputs = crossvalidation_ensemble(nFolds, inputs, targets, models, ...
                                             preprocessors, trainer, sampler)
                                          

dataSize = size(inputs, 2);
ensembleSize = length(models);
outputSize = length(models{1}.output(preprocessors{1}.transform(inputs(:,1))));
outputs = zeros(ensembleSize*outputSize, dataSize);
hold_outs = cross_validate_partition(dataSize, nFolds);

for foldIdx = 1:nFolds
   validIdx = hold_outs{foldIdx};
   trainIdx = setdiff(1:dataSize, validIdx);
   
   for i = 1:ensembleSize
      
      fprintf('\n\n Fold %d Model %d training started.\n', foldIdx, i);
      % Set model
      trainer.model = models{i};
      
      % Sample from training data and apply preprocessor to set dataManager
      sampleIdx = sampler.sample(trainIdx);
      trainer.dataManager.trainingInputs = preprocessors{i}.transform(inputs(:, sampleIdx));
      trainer.dataManager.trainingTargets = targets(:, sampleIdx);
      trainer.dataManager.validationInputs = preprocessors{i}.transform(inputs(:, validIdx));
      trainer.dataManager.validationTargets = targets(:, validIdx);
      
      % Reset trainer and train model
      trainer.reset();
      trainer.train();
      
      % Make prediction for validation set
      startIdx = (i-1)*outputSize + 1;
      stopIdx = i*outputSize;
      outputs(startIdx:stopIdx, validIdx) = ...
         gather(trainer.model.output(trainer.dataManager.validationInputs));
   end
end

end

