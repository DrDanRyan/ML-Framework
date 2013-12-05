classdef FlattenLayer < NoParamsLayer
   % Takes multiple dimension input and flattens to a single dimension
   
   methods
      function [y, ffExtras] = feed_forward(~, x)
         N = size(x, 2);
         D = ndims(x);
         y = permute(x, [1, 3:D, 2]);
         y = reshape(y, [], N);
         ffExtras = [];
      end
      
      function [grad, dLdx] = backprop(~, x, ~, dLdy)
         grad = [];
         D = ndims(x);
         xSize = size(x);
         shapeVec = [xSize(1), xSize(3:D), xSize(2)];
         dLdx = reshape(dLdy, shapeVec);
         dLdx = permute(dLdx, [1, D, 2:D-1]);
      end
   end
   
end

