classdef ConvHiddenLayer < HiddenLayer
   
   properties
      convLayer
      noParamsLayer
      poolingLayer
   end
   
   methods
      function y = feed_forward(obj, x, isSave)
         if nargin < 3
            isSave = false;
         end
         y = obj.convLayer.feed_forward(x);
         y = obj.noParamsLayer.feed_forward(y, isSave);
         y = obj.poolingLayer.pool(y, isSave);
      end
      
      function [grad, dLdx] = backprop(obj, x, ~, dLdy)
         dLdx = obj.poolingLayer.unpool(dLdy);
         dLdx = obj.noParamsLayer.backprop(dLdx);
         [grad, dLdx] = obj.convLayer.backprop(x, dLdx);
      end
      
      function init_params(obj)
         obj.convLayer.init_params(obj);
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
   end
end

