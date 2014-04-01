classdef LPPoolingLayer < CompositeHigherLayer & matlab.mixin.Copyable & ...
                          ParamsFunctions
   % An L-p pooling layer with learnable exponent p. Performs pooling across 3rd
   % dimension of input.
   
   properties
      % params = {phat} ~ L x 1, where p = 1 + exp(phat)
      L % number of hidden unit groups; this is set upon first input observation
      dydphat
      dydx
   end
   
   methods
      function obj = LPPoolingLayer(varargin)
         obj = obj@ParamsFunctions(varargin{:});
      end
      
      function init_params(obj)
         if isempty(obj.initScale)
            obj.initScale = .5;
         end
         obj.params{1} = obj.initScale*obj.gpuState.randn(obj.L, 1);
      end
      
      function y = feed_forward(obj, x, isSave)
         if isempty(obj.L)
            obj.L = size(x, 1);
            obj.init_params();
         end
         
         % Scale x before applying exponents to prevent underflow/overflow
         p = 1 + exp(obj.params{1});
         absx = abs(x);
         maxVals = max(absx, [], 3);
         absx = bsxfun(@rdivide, absx, maxVals);

         xp = bsxfun(@power, absx, p);
         sum_xp = sum(xp, 3);
         yScaled = bsxfun(@power, sum_xp, 1./p);
         y = maxVals.*yScaled;
         if nargin == 3 && isSave
            temp = bsxfun(@power, sum_xp, (p-1)./p);
            obj.dydx = ...
               bsxfun(@rdivide, sign(x).*bsxfun(@power, absx, p-1), temp);
            xp_logabsx = xp.*log(absx);
            xp_logabsx(isnan(xp_logabsx)) = 0; % covers the cases where absx==0
            temp = maxVals.*(sum(xp_logabsx, 3)./temp - yScaled.*log(yScaled));
            obj.dydphat = bsxfun(@times, temp, (p-1)./p);
         end
      end
      
      function [grad, dLdx] = backprop(obj, dLdy)
         dLdx = bsxfun(@times, obj.dydx, dLdy);
         obj.dydx = [];
         grad = {mean(obj.dydphat.*dLdy, 2)};
         obj.dydphat = [];
      end
   end
   
end

