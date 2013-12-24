classdef SparseFilteringLoss < LossFunction
   % Assumes output is non-negative. Replaces all zero values with 1e-8 to
   % prevent degeneracies.

   methods
      function dLdy = dLdy(~, y, ~)
         y = max(y, 1e-8); % set minimum value for y
         rowNorms = sqrt(sum(y.*y, 2));
         yRowNormed = bsxfun(@rdivide, y, rowNorms);
         colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
         F = bsxfun(@rdivide, yRowNormed, colNorms);
         
         % Backprop
         if isa(F, 'gpuArray')
            dLdy = gpuArray.ones(size(F), 'single');
         else
            dLdy = ones(size(F));
         end
         
         dLdy = bsxfun(@rdivide, dLdy, colNorms) ...
                     - bsxfun(@times, F, sum(dLdy.*yRowNormed, 1)./(colNorms.*colNorms));       
         dLdy = bsxfun(@rdivide, dLdy, rowNorms) ...
                     - bsxfun(@times, yRowNormed, sum(dLdy.*y, 2)./(rowNorms.*rowNorms));
                  
         if check_nan(dLdy)
            keyboard();
         end
      end
      
      function loss = compute_loss(~, y, ~)
         y = max(y, 1e-8);
         rowNorms = sqrt(sum(y.*y, 2));
         yRowNormed = bsxfun(@rdivide, y, rowNorms);
         colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
         F = bsxfun(@rdivide, yRowNormed, colNorms);
         loss = sum(abs(F(:)));
      end
   end
   
end

