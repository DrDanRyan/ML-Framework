classdef HiddenLayer < matlab.mixin.Copyable
   % Defines the HiddenLayer interface 
   
   properties (Abstract)
      isLocallyLinear
   end

   methods (Abstract)
      [grad, dLdx, Dy] = backprop(obj, x, y, dLdy)
      y = feed_forward(obj, x)
      value = compute_z(obj, x)
      value = compute_Dy(obj, x, y) % derivative of transfer function
      value = compute_D2y(obj, x, y) % second derivatie of transfer function
      push_to_GPU(obj)
      gather(obj)
      increment_params(obj, delta_params)
      init_params(obj)
   end
   
end

