classdef JacobianPenalty
   
   properties
      C
   end
   
   methods
      function obj = JacobianPenalty(C)
         obj.C = C;
      end
      
      function value = penalty_gradient(obj, layer, dydz)
         dydzSqMean = mean(dydz.*dydz, 2);
         value = obj.C*bsxfun(@times, layer.params{1}, dydzSqMean);
      end
   end
   
end

