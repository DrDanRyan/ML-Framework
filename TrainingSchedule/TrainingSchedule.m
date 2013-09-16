classdef TrainingSchedule < matlab.mixin.Copyable
   % <Interface> Adjusts StepCalculator parameters and determines when 
   % training should terminate.
    
   properties (Abstract)
      params
   end
   
   methods (Abstract)
      isContinue = update(obj, trainer, trainingLoss, validationLoss)
      reset(obj)
   end
   
end

