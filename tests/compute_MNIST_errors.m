function nErrors = compute_MNIST_errors(model, dataManager)
[~, predictions] = max(model.output(dataManager.validationData{1}));
[~, actual] = max(dataManager.validationData{2});
nErrors = sum(predictions~=actual);
end

