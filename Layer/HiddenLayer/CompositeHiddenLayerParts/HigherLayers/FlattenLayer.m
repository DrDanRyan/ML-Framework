classdef FlattenLayer < matlab.mixin.Copyable
   % Takes multiple dimension input and flattens to a single dimension
   
   properties
      D
      xSize
   end
   
   methods
      function y = feed_forward(obj, x, ~)
         obj.xSize = size(x);
         obj.D = ndims(x);
         y = permute(x, [1, 3:obj.D, 2]);
         y = reshape(y, [], obj.xSize(2));
      end
      
      function dLdx = backprop(obj, dLdy)
         shapeVec = [obj.xSize(1), obj.xSize(3:end), obj.xSize(2)];
         dLdx = reshape(dLdy, shapeVec);
         dLdx = permute(dLdx, [1, obj.D, 2:obj.D-1]);
      end
   end
   
end

