classdef MeanSquaredError < LossFunction
   % A LossFunction for mean squared error. Works for output with an arbitrary
   % number of dimensions.
   
   methods
      function dLdy = dLdy(~, y, t)
         dLdy = 2*(y - t);
      end
      
      function loss = compute_loss(~, y, t)
         loss = sum((y(:) - t(:)).^2)/size(y, 2);
      end
   end
   
end

