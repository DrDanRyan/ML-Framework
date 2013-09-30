classdef LinearHiddenLayer < StandardHiddenLayer
   % A simple linear layer. Useful for constructing a MaxoutAutoEncoder.
   
   properties
      nonlinearity = @(x) x; % not actually used, need to define to inherit from StandardHiddenLayer
   end
   
   methods
      function y = feed_forward(obj, x)
         y = bsxfun(@plus, obj.params{1}*x, obj.params{2});
      end
      
      function value = dydz(obj, y)
         value = obj.gpuState.ones(size(y));
      end
      
      
   end
   
end

