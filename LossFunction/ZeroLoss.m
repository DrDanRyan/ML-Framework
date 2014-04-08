classdef ZeroLoss < LossFunction
   % A loss function that always returns zero (useful for unsupervised training
   % with only regularization terms, e.g. sparse filtering)
   
   methods
      function dLdy = dLdy(~,y,~)
         dLdy = 0.*y; % is using a gpuState better?
      end
      
      function loss = compute_loss(~,~,~)
         loss = 0;
      end
   end
   
end

