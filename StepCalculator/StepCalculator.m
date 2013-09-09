classdef StepCalculator < matlab.mixin.Copyable
   % This defines the StepCalculator interface
   
   methods (Abstract)
      take_step(obj, x, t, model, params)
      reset(obj)
   end
   
end

