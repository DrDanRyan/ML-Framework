classdef ReluNoParamsLayer < NoParamsHiddenLayer
   
   properties
      isLocallyLinear = true
   end
   
   methods
      function [y, ffExtras] = feed_forward(~, x)
         y = max(0, x);
         ffExtras = [];
      end   
      
      function value = compute_Dy(~, ~, y)
         if isa(y, 'gpuArray')
            value = single(y > 0);
         else
            value = y > 0; % conversion can happen automatically later
         end
      end
   end
end

