classdef NaNMeanSquaredError < LossFunction
   
   methods
      function dLdy = dLdy(obj, y, t)
         dLdy = y - t;
         dLdy(isnan(t)) = 0;
      end
      
      function loss = compute_loss(obj, y, t)
         loss = .5*nansum(nansum((y - t).^2))/sum(sum(isnan(t)));
      end
   end
   
end

