function error_rate = training_binary_error(model, dataManager, sampleSize)
if sampleSize < dataManager.trainingSize
   permvec = randperm(dataManager.trainingSize, sampleSize);
   y = model.output(dataManager.trainingData{1}(:,permvec));
   t = dataManager.trainingData{end}(:,permvec);
else
   y = model.output(dataManager.trainingData{1});
   t = dataManager.trainingData{end};
end
error_rate = binary_error(y, t);
end

