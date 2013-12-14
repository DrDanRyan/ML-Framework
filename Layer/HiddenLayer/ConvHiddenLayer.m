classdef ConvHiddenLayer < HiddenLayer
   
   properties
      convLayer
      noParamsLayer
      poolingLayer
      flattenLayer
   end
   
   properties (Dependent)
      params
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         if nargin < 3
            isSave = false;
         end
         y = obj.convLayer.feed_forward(x);
         y = obj.noParamsLayer.feed_forward(y, isSave);
         y = obj.poolingLayer.pool(y, isSave);
         if ~isempty(obj.flattenLayer)
            y = obj.flattenLayer.feed_forward(y, isSave);
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         if ~isempty(obj.flattenLayer)
            dLdx = obj.flattenLayer.backprop(dLdy);
            dLdx = obj.poolingLayer.unpool(dLdx);
         else
            dLdx = obj.poolingLayer.unpool(dLdy);
         end
         dLdx = obj.noParamsLayer.backprop(dLdx);
         [grad, dLdx] = obj.convLayer.backprop(x, dLdx);
      end
      
      function params = get.params(obj)
         if ~isempty(obj.convLayer)
            params = obj.convLayer.params;
         end
      end
      
      function set.params(obj, newParams)
         obj.convLayer.params = newParams;
      end
      
      function init_params(obj)
         obj.convLayer.init_params();
      end
      
      function increment_params(obj, delta)
         obj.convLayer.increment_params(delta);
      end
      
      function push_to_GPU(obj)
         obj.convLayer.push_to_GPU();
      end
      
      function gather(obj)
         obj.convLayer.gather();
      end
      
      function objCopy = copy(obj)
         objCopy = ConvHiddenLayer();
         objCopy.convLayer = obj.convLayer.copy();
         objCopy.noParamsLayer = obj.noParamsLayer.copy();
         objCopy.poolingLayer = obj.poolingLayer.copy();
         if ~isempty(obj.flattenLayer)
            objCopy.flattenLayer = obj.flattenLayer.copy();
         end
      end
   end
end

