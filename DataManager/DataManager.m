classdef DataManager < matlab.mixin.Copyable
   % This defines the DataManager interface
   
   methods (Abstract)
      [x, t, endOfEpochFlag] = next_batch(obj)
      [inputs, targets] = get_validation_data(obj)
      [inputs, targets] = get_training_data(obj)
      reset(obj)
   end
   
end

