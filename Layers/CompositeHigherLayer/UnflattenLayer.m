classdef UnflattenLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % Takes 1D feature vector and shapes it into NDim Topological feature vector
   
   properties
      shape
   end
   
   methods
      function obj = UnflattenLayer(shape)
         obj.shape = shape;
      end
      
      function y = feed_forward(obj, x, ~)
         N = size(x, 2);
         y = reshape(x, [obj.shape(1), N, obj.shape(2:end)]);
      end
      
      function dLdx = backprop(~, dLdy)
         N = size(dLdy, 2);
         ndims = ndim(dLdy);
         dLdx = permute(dLdy, [1, 3:ndims, 2]);
         dLdx = reshape(dLdx, [], N);
      end
         
   end
   
end

