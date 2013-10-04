classdef ComboOutputLayer < OutputLayer
   % This class combines a HiddenLayer object with a LossFunction object to
   % act as an OutputLayer.
   
   properties
      hiddenLayer
      lossFunction
   end
   
   methods
      function obj = ComboOutputLayer(hiddenLayer, lossFunction)
         obj.hiddenLayer = hiddenLayer;
         obj.lossFunction = lossFunction;
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t, isAveraged)
         if nargin < 4
            isAveraged = true;
         end
         y = obj.hiddenLayer.feed_forward(x);
         dLdy = obj.lossFunction.dLdy(y, t);
         [grad, dLdx] = obj.hiddenLayer.backprop(x, y, dLdy, isAveraged);
      end
      
      function loss = compute_loss(obj, y, t)
         loss = obj.lossFunction.compute_loss(y, t);
      end
      
      function y = feed_forward(obj, x)
         y = obj.hiddenLayer.feed_forward(x);
      end
      
      function push_to_GPU(obj)
         obj.hiddenLayer.push_to_GPU();
      end
      
      function gather(obj)
         obj.hiddenLayer.gather();
      end
      
      function increment_params(obj, delta_params)
         obj.hiddenLayer.increment_params(delta_params);
      end
      
      function init_params(obj)
         obj.hiddenLayer.init_params();
      end
   end
   
end

