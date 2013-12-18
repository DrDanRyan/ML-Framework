classdef StochasticPooling2DLayer < PoolingLayer
   
   properties
      inputRows
      inputCols
      poolRows
      poolCols
      winners
      gpuState
   end
   
   methods
      function obj = StochasticPooling2DLayer(poolRows, poolCols)
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
            obj.winners = obj.gpuState.false(nF, N, obj.poolRows*obj.poolCols, outRows*outCols);
         end
         
         k = 1;
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               xSeg = x(:,:, rowStart:rowEnd, colStart:colEnd);
               xSeg = reshape(xSeg, nF, N, []);
               probs = bsxfun(@rdivide, xSeg, sum(xSeg, 3));
               sample = multinomial_sample(probs, 3);
               xPool(:,:,i,j) = sum(sample.*xSeg, 3);
               if nargin == 3 && isSave
                  squareSize = size(xSeg, 3);
                  obj.winners(:,:,1:squareSize,k) = logical(sample);
                  k = k+1;
               end
            end
         end  
      end
      
      function yUnpool = unpool(obj, y)
         [nF, N, outRows, outCols] = size(y);
         if isa(obj.winners, 'gpuArray')
            obj.winners = single(obj.winners); % convert from logicals
         end
         
         yUnpool = obj.gpuState.zeros(nF, N, obj.inputRows, obj.inputCols);
         k = 1;
         for i = 1:outRows
            for j = 1:outCols
               rowStart = (i-1)*obj.poolRows + 1;
               rowEnd = min(rowStart + obj.poolRows - 1, obj.inputRows);
               colStart = (j-1)*obj.poolCols + 1;
               colEnd = min(colStart + obj.poolCols - 1, obj.inputCols);
               yUnpool(:,:,rowStart:rowEnd, colStart:colEnd) = ...
                  reshape(bsxfun(@times, y(:,:,i,j), obj.winners(:,:,:,k)), ...
                              nF, N, rowEnd-rowStart+1, colEnd-colStart+1);
               k = k+1;
            end
         end
         obj.winners = [];
      end
   end
end

