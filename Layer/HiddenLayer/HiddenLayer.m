classdef HiddenLayer < matlab.mixin.Copyable
   % Defines the HiddenLayer interface 
   
   properties (Abstract)
      % need to implement for CAE and ManifoldTangentClassifier
      isLocallyLinear 
   end

   methods (Abstract)
      [grad, dLdx, y] = backprop(obj, x, y, z, dLdy)
      [y, z] = feed_forward(obj, x)
      value = compute_z(obj, x)
      value = compute_Dy(obj, z, y) % derivative of transfer function
      value = compute_D2y(obj, z, y, Dy) % second derivatie of transfer function
      push_to_GPU(obj)
      gather(obj)
      increment_params(obj, delta_params)
      init_params(obj)
   end
   
end

