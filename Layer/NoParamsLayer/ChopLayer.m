classdef ChopLayer < NoParamsLayer
   % Takes multiple dimension input and chops into multiple examples,
   % keeping first dimension as the input dimension (useful for
   % convolutional sparse filtering)
   
   properties
      xSize
   end
   
   methods
      function y = feed_forward(obj, x, ~)
         obj.xSize = size(x);
         y = reshape(x, obj.xSize(1), []);
      end
      
      function dLdx = backprop(obj, dLdy)
         dLdx = reshape(dLdy, obj.xSize);
      end
   end
   
end


