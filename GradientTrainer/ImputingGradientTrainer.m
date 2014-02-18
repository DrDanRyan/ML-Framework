classdef ImputingGradientTrainer < GradientTrainer
   
   methods
      function isContinue = update(obj)
         isContinue = update@GradientTrainer(obj);
         obj.dataManager.update_imputed_data(obj.model.imputedData);
      end
   end
   
end

