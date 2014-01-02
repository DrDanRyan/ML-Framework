classdef SparseFilteringLoss < LossFunction
   % Assumes output is non-negative. Fudges all y values by adding eps
   % to prevent degenericies arising from all examples of a feature == 0

   properties
      eps
   end
   
   methods
      function obj = SparseFilteringLoss(eps)
         if nargin < 1
            eps = 1e-4;
         end
         obj.eps = eps;
      end
      
      function dLdy = dLdy(obj, y, ~)
         y = y + obj.eps;
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
      end
      
      function loss = compute_loss(obj, y, ~)
         y = y + obj.eps;
         rowNorms = sqrt(sum(y.*y, 2));
         yRowNormed = bsxfun(@rdivide, y, rowNorms);
         colNorms = sqrt(sum(yRowNormed.*yRowNormed, 1));
         F = bsxfun(@rdivide, yRowNormed, colNorms);
         loss = sum(F(:));
      end
      
   end
end

