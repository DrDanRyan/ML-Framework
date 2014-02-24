classdef TRecHiddenLayer < matlab.mixin.Copyable & ParamsFunctions
   
   properties
      inputSize
      outputSize
      theta
      dydx
   end
   
   methods
      function obj = TRecHiddenLayer(inputSize, outputSize, varargin)
         obj = obj@ParamsFunctions(varargin{:});
         p = inputParser;
         p.KeepUnmatched = true;
         p.addParamValue('theta', 1);
         parse(p, varargin{:});
         
         obj.inputSize = inputSize;
         obj.outputSize = outputSize;
         obj.theta = p.Results.theta;
         obj.init_params();
      end
      
      function init_params(obj)
         obj.params{1} = matrix_init(obj.outputSize, obj.inputSize, ...
                                     obj.initType, obj.initScale, obj.gpuState);
      end
      
      function y = feed_forward(obj, x, isSave)
         y = obj.params{1}*x;
         cutIdx = y < obj.theta;
         y(cutIdx) = 0;
         
         if nargin == 3 && isSave
            obj.dydx = ~cutIdx;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         N = size(x, 2);
         dLdz = obj.dydx.*dLdy;
         obj.dydx = [];
         grad{1} = dLdz*x'/N;
         dLdx = obj.params{1}'*dLdz;
      end
      
   end
end

