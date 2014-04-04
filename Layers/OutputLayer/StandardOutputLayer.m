classdef StandardOutputLayer < OutputLayer & StandardLayer
   % A base class provide standard functionality for an OutputLayer with normal
   % linear layer followed by elementwise nonlinearity structure. A subclass
   % needs to provide at minimum: feed_forward, compute_dLdz, and compute_loss.
   
   methods
      function obj = StandardOutputLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t)
         [dLdz, y] = obj.compute_dLdz(x, t);
         dLdx = obj.params{1}'*dLdz;
         grad = obj.grad_from_dLdz(x, dLdz);
      end
   end
   
   methods (Abstract)
      [dLdz, y] = compute_dLdz(obj, x, t)
   end
   
end

