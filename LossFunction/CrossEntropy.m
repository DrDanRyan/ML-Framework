classdef CrossEntropy < LossFunction
   % Binary cross entropy loss function
   
   methods
      function dLdy = dLdy(obj, y, t)
         dLdy = (y-t)./(y.*(1 - y));
      end
      
      function loss = compute_loss(obj, y, t)
         % loss = mean(-t.*log(y) - (1-t).*log(1-y));
         loss = (sum(-log(y(t==1))) + sum(-log(1-y(t==0))))/length(t); % more robust to NaN from
                                                                       % 0*(-Inf) when t == y == 0
                                                                       % or t == y == 1
      end
   end
   
end

