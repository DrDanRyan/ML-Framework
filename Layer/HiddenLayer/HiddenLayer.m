classdef HiddenLayer < handle
   % Defines the HiddenLayer interface 
   
   % properties required for CAE and MTC
      % isLocallyLinear  
   % end

   methods (Abstract)
      y = feed_forward(obj, x, isSave)
      [grad, dLdx] = backprop(obj, x, y, dLdy)
      init_params(obj)
      increment_params(obj, delta)
      push_to_GPU(obj)
      gather(obj)
      
      % Required for CAE and MTC:
      % Dy = compute_Dy(obj, x, y)  (derivative of transfer function)
      % D2y = compute_D2y(obj, x, y, Dy)   (only required if isLocallyLinear == false)
   end
end

