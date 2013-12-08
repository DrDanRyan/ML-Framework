classdef PoolingLayer < matlab.mixin.Copyable
   % Defines the interface for PoolingLayers
   
   methods (Abstract)
      xPool = pool(obj, x, isSave)
      yUnpool = unpool(obj, y)
   end
   
end

