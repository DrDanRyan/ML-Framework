classdef SparseFilteringLoss < LossFunction
   
   methods      
      function dLdy = dLdy(~, y, ~)
         rowNorms = sqrt(sum(y.*y, 2));
         yRowNormed = bsxfun(@rdivide, y, rowNorms);
         colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
         F = bsxfun(@rdivide, yRowNormed, colNorms);
         
         % Backprop
         dLdy = sign(F);         
         dLdy = bsxfun(@rdivide, dLdy, colNorms) ...
                     - bsxfun(@times, F, sum(dLdy.*yRowNormed, 1)./(colNorms.*colNorms));       
         dLdy = bsxfun(@rdivide, dLdy, rowNorms) ...
                     - bsxfun(@times, yRowNormed, sum(dLdy.*y, 2)./(rowNorms.*rowNorms));
      end
      
      function loss = compute_loss(~, y, ~)
         rowNorms = sqrt(sum(y.*y, 2));
         yRowNormed = bsxfun(@rdivide, y, rowNorms);
         colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
         F = bsxfun(@rdivide, yRowNormed, colNorms);
         loss = mean(sum(abs(F), 1), 2);
      end
      
   end
end

