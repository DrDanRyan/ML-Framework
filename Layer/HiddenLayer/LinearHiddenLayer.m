classdef LinearHiddenLayer < StandardLayer & HiddenLayer
   % A simple linear layer.
   
   properties
      % isLocallyLinear = true
   end
   
   methods
      function obj = LinearHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x, ~)
         y = obj.compute_z(x);
      end   
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdy);
      end
   end
end

