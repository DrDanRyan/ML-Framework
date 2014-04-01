classdef Conv2DMaxPoolingLayer < CompositeHigherLayer & matlab.mixin.Copyable
   % Performs max pooling of nonoverlapping square regions for each channel in
   % a multi-channel 2D input signal.
   
   properties
      inputRows
      inputCols
      poolRows
      poolCols
      winners
      gpuState
   end
   
   methods
      function obj = Conv2DMaxPoolingLayer(poolRows, poolCols)
         obj.poolRows = poolRows;
         obj.poolCols = poolCols;
         obj.gpuState = GPUState();
      end
      
      function y = feed_forward(obj, x, isSave)
         obj.gpuState.isGPU = isa(x, 'gpuArray');
         [nF, N, obj.inputRows, obj.inputCols] = size(x);
         outRows = ceil(obj.inputRows/obj.poolRows);
         outCols = ceil(obj.inputCols/obj.poolCols);
         y = obj.gpuState.nan(nF, N, outRows, outCols);
         if nargin == 3 && isSave
            obj.winners = obj.gpuState.false(nF, N, obj.inputRows, obj.inputCols);
         end
         
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               samp = x(:,:, rowStart:rowEnd, colStart:colEnd);
               [maxVals, colIdx] = max(samp, [], 4); % colIdx ~ C x N x poolRows
               [y(:,:,i,j), rowIdx] = max(maxVals, [], 3); % rowIdx ~ C x N
               if nargin == 3 && isSave
                  for s = 1:obj.poolCols
                     for r = 1:obj.poolRows
                        obj.winners(:,:,rowStart+r-1, colStart+s-1) = ...
                            colIdx(:,:,r)==s & rowIdx==r;
                     end
                  end
               end
            end
         end  
      end
      
      function dLdx = backprop(obj, dLdy)
         [~, ~, outRows, outCols] = size(dLdy);       
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               dLdx(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                  bsxfun(@times, dLdy(:,:,i,j), ...
                  obj.winners(:,:,rowStart:rowEnd,colStart:colEnd));
            end
         end
         obj.winners = [];
      end
      
   end
end

