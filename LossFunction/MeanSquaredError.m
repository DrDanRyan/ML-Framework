classdef MeanSquaredError < LossFunction
   
   methods
      function dLdy = dLdy(obj, y, t)
         dLdy = 2*(y - t);
      end
      
      function loss = compute_loss(obj, y, t)
         loss = sum((y - t).^2)/size(y, 2);
      end
   end
   
end

