classdef NaNMeanSquaredError < LossFunction
   
   methods
      function dLdy = dLdy(~, y, t)
         dLdy = y - t;
         dLdy(isnan(t)) = 0;
      end
      
      function dLdt = dLdt(~, y, t)
         % Used by ImputingAutoEncoder when learning imputed values
         dLdt = t - y;
      end
      
      function loss = compute_loss(~, y, t)
         loss = .5*nansum(nansum((y - t).^2))/sum(sum(isnan(t)));
      end
   end
   
end

