classdef SparseFilteringLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % A "sparse filtering" layer, see Ngiam2011 "Sparse Filtering". 
   %
   % First, each feature activation across examples is L2 normalized (rows have
   % L2 norm == 1). Then each example is L2 normalized across features (columns
   % are made to have L2 norm == 1). These vectors are the outputs. In
   % addition, an L1 penalty term on the output vector is added to the loss
   % (allowing unsupervised training by secifying a zero loss function).
   % 
   % Works best with something like SoftAbs activation function (that is always 
   % strictly positive).
   
   properties
      penaltyCoeff % penalty coefficient for mean L1 norm of output values
      
      % Stored for backprop purposes
      x
      y
      rowNorms
      colNorms
      
   end
   
   methods      
      function obj = SparseFilteringLayer(penaltyCoeff)
         if nargin < 1
            penaltyCoeff = 1;
         end
         
         obj.penaltyCoeff = penaltyCoeff;
      end
      
      function y = feed_forward(obj, x, isSave)
         
         % Row normalization
         rowNorms = sqrt(sum(x.*x, 2));
         xRowNormed = bsxfun(@rdivide, x, rowNorms);
         
         % Column normalization
         colNorms = sqrt(sum(xRowNormed.*xRowNormed, 1));
         y = bsxfun(@rdivide, xRowNormed, colNorms);
         
         if nargin == 3 && isSave
            obj.x = x;
            obj.y = y;
            obj.rowNorms = rowNorms;
            obj.colNorms = colNorms;
         end
      end
      
      function dLdx = backprop(obj, dLdy)
         
         % Add in penalty from L1 output norm
         dLdx = dLdy + obj.penaltyCoeff*sign(obj.y);
         
         % compute normalized x (because it wasn't stored)
         xRowNormed = bsxfun(@rdivide, obj.x, obj.rowNorms);
         
         % Backprop through column normalization
         dLdx = bsxfun(@rdivide, dLdx, obj.colNorms) ...
                     - bsxfun(@times, obj.y, sum(dLdx.*xRowNormed, 1)./...
                     (obj.colNorms.*obj.colNorms));       
                  
         % Backprop through row normalization
         dLdx = bsxfun(@rdivide, dLdx, obj.rowNorms) ...
                     - bsxfun(@times, xRowNormed, sum(dLdx.*obj.x, 2)./...
                     (obj.rowNorms.*obj.rowNorms));
      end
     
   end
end

