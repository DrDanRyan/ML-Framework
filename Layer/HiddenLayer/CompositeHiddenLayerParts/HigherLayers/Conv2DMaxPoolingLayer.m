classdef Conv2DMaxPoolingLayer < matlab.mixin.Copyable
   
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
            obj.winners = obj.gpuState.zeros(nF, N, obj.inputRows, obj.inputCols);
         end
         
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               samp = x(:,:, rowStart:rowEnd, colStart:colEnd);
               y(:,:,i,j) = max(max(samp, [], 3), [], 4);
               if nargin == 3 && isSave
                  obj.winners(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                                                      bsxfun(@eq, samp, y(:,:,i,j));
               end
            end
         end  
      end
      
      function dLdx = backprop(obj, dLdy)
         [nF, N, outRows, outCols] = size(dLdy);
         if isa(obj.winners, 'gpuArray')
            obj.winners = single(obj.winners);
         end
         
         dLdx = obj.gpuState.zeros(nF, N, obj.inputRows, obj.inputCols);
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               dLdx(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                  bsxfun(@times, dLdy(:,:,i,j), obj.winners(:,:,rowStart:rowEnd,colStart:colEnd));
            end
         end
         obj.winners = [];
      end
      
   end
end

