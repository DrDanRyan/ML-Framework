classdef InspectorOutputLayer < OutputLayer
   
   properties
      hiddenIdx
      gpuState
   end
   
   methods
      function obj = InspectorOutputLayer(hiddenIdx, gpuState)
         obj.hiddenIdx = hiddenIdx;
         obj.gpuState = gpuState;
      end
      
      
      function y = feed_forward(obj, x)
         y = -x(obj.hiddenIdx);
      end
      
      
      function [grad, dLdx, y] = backprop(obj, x, ~)
         grad = [];         
         dLdx = obj.gpuState.zeros(size(x));
         dLdx(obj.hiddenIdx) = -1;
         y = -x(obj.hiddenIdx);
      end
      
      
      function loss = compute_loss(obj, y, ~)
         loss = y(obj.hiddenIdx);
      end
      
      
      function init_params(obj)
         % pass
      end
      
      
      function increment_params(obj, delta)
         % pass
      end
      
      
      function push_to_GPU(obj)
         % pass
      end
      
      
      function gather(obj)
         % pass
      end
         
   end
   
end

