classdef StepCalculator < matlab.mixin.Copyable
   % <Interface> Uses model.gradient(x,t) and params to determine next step
   % direction and size. Calls model.increment_params(step) to update the
   % model accordingly.
   
   methods (Abstract)
      take_step(obj, batch, model, params)
      reset(obj)
   end
   
end

