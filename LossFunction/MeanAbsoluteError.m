classdef MeanAbsoluteError < LossFunction
   
   methods
      function dLdy = dLdy(~, y, t)
         dLdy = sign(y - t);
      end
      
      function loss = compute_loss(~, y, t)
         loss = mean(abs((y(:) - t(:))));
      end
   end
   
end

