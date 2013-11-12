function loss = MNIST_validLossFunction(model, dataManager)
y = model.output(dataManager.validationData{1});
t = dataManager.validationData{2};
loss = compute_error_rate(y, t);
end

