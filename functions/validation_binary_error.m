function error_rate = validation_binary_error(model, dataManager)
y = model.output(dataManager.validationData{1});
t = dataManager.validationData{end};
error_rate = binary_error(y, t);
end

