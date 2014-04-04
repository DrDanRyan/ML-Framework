classdef NaNMeanSquaredError < LossFunction
   % A LossFunction for mean squared error but ignores targets that are NaN.
   
   methods
      function dLdy = dLdy(~, y, t)
         dLdy = 2*(y - t);
         dLdy(isnan(t)) = 0;
      end
      
      function dLdt = dLdt(~, y, t)
         % Used by ImputingAutoEncoder when learning imputed values
         dLdt = 2*(t - y);
      end
      
      function loss = compute_loss(~, y, t)
         loss = nanmean((y(:) - t(:)).^2);
      end
   end
   
end

