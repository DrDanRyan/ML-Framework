function ensemble = train_ensemble(inputs, targets, models, preprocessors, trainer, sampler)
   ensembleSize = length(models); % number of models in the ensemble
   dataSize = size(inputs, 2); % number of examples in training data
   outputSize = length(models{1}.output(preprocessors{1}.transform(inputs(:,1))));
   
   for i = 1:ensembleSize
      trainer.model = models{i};
      trainIdx = sampler.sample(1:dataSize);
      validIdx = setdiff(1:dataSize, trainIdx);
      trainer.dataManager.trainingInputs = preprocessors{i}.transform(inputs(:, trainIdx));
      trainer.dataManager.trainingTargets = targets(:, trainIdx);
      trainer.dataManager.validationInputs = preprocessors{i}.transform(inputs(:, validIdx));
      trainer.dataManager.validationTargets = targets(:, validIdx);
      trainer.reset();
      trainer.train();
   end

   ensemble = Ensemble(models, preprocessors, outputSize);
end

