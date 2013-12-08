classdef StochasticPooling1DLayer < PoolingLayer
   
   properties
      poolSize
      inputSize
      winners
   end
   
   methods
      function xPool = pool(obj, x, isSave)
         % if isSave is false, use linear combination, else sample based on
         % multinomial probabilities
      end
      
      function yUnpool = unpool(obj, y)
         
      end
   end
   
end

