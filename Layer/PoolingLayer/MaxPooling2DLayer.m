classdef MaxPooling2DLayer < PoolingLayer
   
   properties
      inputRows
      inputCols
      poolRows
      poolCols
      winners
      gpuState
   end
   
   methods
      function obj = MaxPooling2DLayer(poolRows, poolCols)
         obj.poolRows = poolRows;
         obj.poolCols = poolCols;
         obj.gpuState = GPUState();
      end
      
      function xPool = pool(obj, x, isSave)
         obj.gpuState.isGPU = isa(x, 'gpuArray');
         [nF, N, obj.inputRows, obj.inputCols] = size(x);
         outRows = ceil(obj.inputRows/obj.poolRows);
         outCols = ceil(obj.inputCols/obj.poolCols);
         xPool = obj.gpuState.nan(nF, N, outRows, outCols);
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
               xPool(:,:,i,j) = max(max(samp, [], 3), [], 4);
               if nargin == 3 && isSave
                  obj.winners(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                                                      bsxfun(@eq, samp, xPool(:,:,i,j));
               end
            end
         end  
      end
      
      function yUnpool = unpool(obj, y)
         [nF, N, outRows, outCols] = size(y);
         if isa(obj.winners, 'gpuArray')
            obj.winners = single(obj.winners);
         end
         
         yUnpool = obj.gpuState.zeros(nF, N, obj.inputRows, obj.inputCols);
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               yUnpool(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                  bsxfun(@times, y(:,:,i,j), obj.winners(:,:,rowStart:rowEnd,colStart:colEnd));
            end
         end
         obj.winners = [];
      end
   end
end

