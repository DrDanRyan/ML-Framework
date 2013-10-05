classdef DataManager < matlab.mixin.Copyable
   % This defines the DataManager interface
   
   properties
      trainingInputs
      trainingTargets
      validationInputs
      validationTargets
   end
   
   methods (Abstract)
      [x, t, endOfEpochFlag] = next_batch(obj)
      reset(obj)
   end
   
end

