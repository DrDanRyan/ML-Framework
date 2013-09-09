classdef Reporter < matlab.mixin.Copyable
   % Defines the Reporter interface
   
   methods (Abstract)
      update(obj, trainingLoss, validationLoss)
      reset(obj)
   end
   
end

