classdef ImputingTrainer < Trainer
   % A Trainer that passes imputed missing data from model back to
   % dataManager after each update. Use this class if your model refines
   % estimates for missing data in each batch as it learns.
   
   methods
      function isContinue = update(obj)
         isContinue = update@GradientTrainer(obj);
         obj.dataManager.update_imputed_data(obj.model.imputedData);
      end
   end
   
end

