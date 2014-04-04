classdef MeanAbsoluteError < LossFunction
   % A LossFunction for mean absolute error. Works for outputs that are
   % arbitrary dimension.
   
   methods
      function dLdy = dLdy(~, y, t)
         dLdy = sign(y - t);
      end
      
      function loss = compute_loss(~, y, t)
         loss = sum(abs((y(:) - t(:))))/size(y, 2);
      end
   end
   
end

