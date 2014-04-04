classdef CrossEntropy < LossFunction
   % Binomial cross-entropy loss function. This object will fail if y == 0 or y
   % == 1 (in which case cancellation with dydz in the output layer should be
   % ocurring for logistic units).
   
   methods
      function dLdy = dLdy(~, y, t)
         dLdy = (y-t)./(y.*(1 - y));
      end
      
      function loss = compute_loss(~, y, t)
         loss = mean(-t.*log(y) - (1-t).*log(1-y));
      end
   end
   
end

