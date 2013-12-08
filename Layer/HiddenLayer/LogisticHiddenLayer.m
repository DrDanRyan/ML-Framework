classdef LogisticHiddenLayer < StandardLayer
   
   properties
      Dy
      % isLocallyLinear = false
   end
   
   methods
      function obj = LogisticHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@StandardLayer(inputSize, outputSize, varargin{:});
      end
      
      function y = feed_forward(obj, x, isSave)
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
         if nargin == 3 && isSave
            obj.Dy = exp(-z).*y.*y;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         dLdz = obj.Dy.*dLdy;
         obj.Dy = [];
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdz);
      end
      
%       function value = compute_D2y(~, ~, y, Dy)
%          value = Dy.*(1-2*y);
%       end
   end
end

