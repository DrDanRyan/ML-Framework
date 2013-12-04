classdef HiddenLayer < matlab.mixin.Copyable
   % Defines the HiddenLayer interface 
   
   % properties required for CAE and MTC
      % isLocallyLinear  
   % end

   methods (Abstract)
      [y, ffExtras] = feed_forward(obj, x)
      [grad, dLdx, Dy] = backprop(obj, x, y, ffExtras, dLdy)
      init_params(obj)
      increment_params(obj, delta)
      push_to_GPU(obj)
      gather(obj)
      
      % Required for CAE and MTC:
      % value = compute_Dy(obj, ffExtras, y)  (derivative of transfer function)
      % value = compute_D2y(obj, ffExtras, y, Dy)   (only required if isLocallyLinear == false)
   end
end

