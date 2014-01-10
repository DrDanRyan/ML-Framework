classdef MeanSquaredError < LossFunction
   
   methods
      function dLdy = dLdy(obj, y, t)
         dLdy = y - t;
      end
      
      function loss = compute_loss(obj, y, t)
         loss = .5*mean((y(:) - t(:)).^2);
      end
   end
   
end

