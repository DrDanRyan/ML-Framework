function error_rate = validation_multiclass_error(model, dataManager)
y = model.output(dataManager.validationData{1});
t = dataManager.validationData{end};
error_rate = multiclass_error(y, t);
end

