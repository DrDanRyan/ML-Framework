classdef OutputLayer < matlab.mixin.Copyable
   % Defines the OutputLayer interface (note some methods are provided by
   % the Layer superclass)
   
   % properties required for MTC
      % isLocallyLinear
      % isDiagonalDy
   % end
   
   methods (Abstract)
      y = feed_forward(x)
      [grad, dLdx, y] = backprop(obj, x, t)
      loss = compute_loss(obj, y, t)
      init_params(obj)
      increment_params(obj, delta)
      push_to_GPU(obj)
      gather(obj)
      
      % Required for MTC
      % Dy = compute_Dy(obj, z, y)
      % D2y = compute_D2y(obj, z, y, Dy) (only required if isLocallyLinear == false)
   end   
end