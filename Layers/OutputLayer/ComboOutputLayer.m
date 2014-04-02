classdef ComboOutputLayer < OutputLayer
   % This class combines a HiddenLayer object with a LossFunction object to
   % act as an OutputLayer.
   
   properties
      hiddenLayer
      lossFunction
   end
   
   properties (Dependent)
      params
   end
   
   methods
      function obj = ComboOutputLayer(hiddenLayer, lossFunction)
         obj.hiddenLayer = hiddenLayer;
         obj.lossFunction = lossFunction;
      end
      
      function params = get.params(obj)
         params = obj.hiddenLayer.params;
      end
      
      function set.params(obj, new_params)
         obj.hiddenLayer.params = new_params;
      end
      
      function [grad, dLdx, y] = backprop(obj, x, t)
         y = obj.hiddenLayer.feed_forward(x, true);
         dLdy = obj.lossFunction.dLdy(y, t);
         [grad, dLdx] = obj.hiddenLayer.backprop(x, y, dLdy);
      end
      
      function Dy = compute_Dy(obj, x, y)
         Dy = obj.hiddenLayer.compute_Dy(x, y);
      end
      
      function D2y = compute_D2y(obj, x, y, Dy)
         D2y = obj.hiddenLayer.compute_D2y(x, y, Dy);
      end
      
      function loss = compute_loss(obj, y, t)
         loss = obj.lossFunction.compute_loss(y, t);
      end
      
      function y = feed_forward(obj, x, isSave)
         if nargin < 3
            isSave = false;
         end
         y = obj.hiddenLayer.feed_forward(x, isSave);
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
      
      function objCopy = copy(obj)
         objCopy = ComboOutputLayer(obj.hiddenLayer.copy(), ...
                                    obj.lossFunction.copy());
      end
   end
   
end

