classdef StepCalculator < matlab.mixin.Copyable
   % This defines the StepCalculator interface.
   
   methods (Abstract)
      % This function computes a step and calls the model to increment its
      % parameters accordingly. The params input is passed from the
      % ParameterSchedule object in the Trainer (if there is any) and are
      % parameters for the StepCalculator (not the model), e.g. learning rate
      % and momentum.
      take_step(obj, batch, model, params)
      
      reset(obj)
   end
   
end

