classdef Conv2DMaxMaskingLayer < matlab.mixin.Copyable
   % Like Conv2DMaxPooling except that it just zeros out non-winners in
   % each group instead of actually pooling. Useful for implementing a 
   % convolutional autoencoder.
   
   properties
      inputRows
      inputCols
      poolRows
      poolCols
      winners
      gpuState
   end
   
   methods
      function obj = Conv2DMaxMaskingLayer(poolRows, poolCols)
         obj.poolRows = poolRows;
         obj.poolCols = poolCols;
         obj.gpuState = GPUState();
      end
      
      function y = feed_forward(obj, x, isSave)
         obj.gpuState.isGPU = isa(x, 'gpuArray');
         [~, ~, obj.inputRows, obj.inputCols] = size(x);
         poolRegionRows = ceil(obj.inputRows/obj.poolRows);
         poolRegionCols = ceil(obj.inputCols/obj.poolCols);
         if nargin == 3 && isSave
            obj.winners = obj.gpuState.false(size(x));
         end
         
         y = obj.gpuState.nan(size(x));
         for i = 1:poolRegionRows
            for j = 1:poolRegionCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               samp = x(:,:, rowStart:rowEnd, colStart:colEnd);
               maxVals = max(max(samp, [], 3), [], 4);
               winners = bsxfun(@eq, samp, maxVals); %#ok<*PROP>
               y(:,:,rowStart:rowEnd, colStart:colEnd) = samp.*winners;
               if nargin == 3 && isSave
                  obj.winners(:,:,rowStart:rowEnd, colStart:colEnd) = winners;
               end
            end
         end  
      end
      
      function dLdx = backprop(obj, dLdy)
         poolRegionRows = ceil(obj.inputRows/obj.poolRows);
         poolRegionCols = ceil(obj.inputCols/obj.poolCols);      
         for i = 1:poolRegionRows
            for j = 1:poolRegionCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               dLdx(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                  bsxfun(@times, dLdy(:,:,rowStart:rowEnd,colStart:colEnd), ...
                                 obj.winners(:,:,rowStart:rowEnd,colStart:colEnd));
            end
         end
         obj.winners = [];
      end
          
   end
end

