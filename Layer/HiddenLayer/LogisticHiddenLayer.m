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
         if nargin < 2
            isSave = false;
         end
         
         z = obj.compute_z(x);
         y = 1./(1 + exp(-z));
         if isSave
            obj.Dy = exp(-z).*y.*y;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, y, dLdy)
         if ~isempty(obj.Dy)
            dLdz = obj.Dy.*dLdy;
            obj.Dy = [];
         else
            dLdz = y.*(1-y).*dLdy;
         end
         [grad, dLdx] = obj.grad_from_dLdz(x, dLdz);
      end
      
%       function value = compute_D2y(~, ~, y, Dy)
%          value = Dy.*(1-2*y);
%       end
   end
end

