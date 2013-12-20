classdef PoolingLayer < matlab.mixin.Copyable
   % Defines the interface for PoolingLayers
   
   methods (Abstract)
      xPool = feed_forward(obj, x, isSave)
      dLdyUnpool = backprop(obj, dLdy)
   end
   
end

