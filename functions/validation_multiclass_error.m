function error_rate = compute_validation_error_rate(model, dataManager)
y = model.output(dataManager.validationData{1});
t = dataManager.validationData{end};
error_rate = multiclass_error(y, t);
end

