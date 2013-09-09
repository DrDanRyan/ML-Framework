classdef TrainingSchedule < matlab.mixin.Copyable
   % This defines the TrainingSchedule interface
    
   properties (Abstract)
      params
   end
   
   methods (Abstract)
      isContinue = update(obj, trainer, trainingLoss, validationLoss)
      reset(obj)
   end
   
end

