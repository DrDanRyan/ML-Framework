classdef LPPoolingLayer2 < matlab.mixin.Copyable & ParamsFunctions
   
   properties
      % params = {phat} ~ L x 1, where p = 1 + exp(phat)
      filterSize
      filterType
      kernel
      stride
      filterMatrix
      inputSize
      outputSize
      dydphat
      dydx
   end
   
   methods
      function obj = LPPoolingLayer2(inputSize, filterSize, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('filterType', 'box');
         p.addParamValue('stride', filterSize);
         parse(p, varargin{:});
         
         obj.inputSize = inputSize;
         obj.filterSize = filterSize;
         obj.filterType = p.Results.filterType;
         obj.stride = p.Results.stride;
         if ~isempty(obj.filterType)
            obj.kernel = compute_1Dfilter_kernel(obj.filterType, obj.filterSize, obj.gpuState);
         end
         
         obj.outputSize = ceil((obj.inputSize - obj.filterSize)/obj.stride) + 1;
         obj.init_params();
      end
      
      function init_params(obj)
         if isempty(obj.initScale)
            obj.initScale = .5;
         end
         obj.params{1} = obj.initScale*obj.gpuState.randn(obj.outputSize, 1);
         obj.filterMatrix = obj.gpuState.zeros(obj.outputSize, obj.inputSize);
         startIdx = 1;
         for i = 1:obj.outputSize
            stopIdx = min(startIdx+obj.filterSize-1, obj.inputSize);
            obj.filterMatrix(i,startIdx:stopIdx) = obj.kernel(1:stopIdx-startIdx+1);
            startIdx = startIdx + obj.stride;
         end
      end
      
      function y = feed_forward(obj, x, isSave)
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
            obj.dydx = bsxfun(@rdivide, sign(x).*bsxfun(@power, absx, p-1), temp);
            xp_logabsx = xp.*log(absx);
            xp_logabsx(isnan(xp_logabsx)) = 0; % covers the cases where absx == 0
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

