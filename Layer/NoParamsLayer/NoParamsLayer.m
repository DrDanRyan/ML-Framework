classdef NoParamsLayer < matlab.mixin.Copyable
   % This defines the interface for layers without parameters (often
   % combined with other bits to form full HiddenLayer or OutputLayer
   % objects)
   
   properties
      Dy
   end
   
   methods (Abstract)
      y = feed_forward(obj, x, isSave)
   end
   
   methods
      function dLdx = backprop(obj, dLdy)
         dLdx = obj.Dy.*dLdy;
         obj.Dy = [];
      end
   end
end

