classdef LossFunction < matlab.mixin.Copyable
   % This abstract class defines the LossFunction interface.
   
   methods (Abstract)
      dLdy = dLdy(obj, y, t)
      loss = compute_loss(obj, y, t)
   end
   
end

