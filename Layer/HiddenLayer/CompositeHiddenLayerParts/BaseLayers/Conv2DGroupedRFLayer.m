classdef Conv2DGroupedRFLayer < GroupedRFLayer
   
   properties
      inputRows % (iR) width of the 2D input signal
      inputCols % (iC) columns of the 2D input signal
      filterRows % (fR)
      filterCols % (fC)
      yRows      % (yR = iR - fR + 1)
      yCols      % (yC = iC - fC + 1)
   end
   
   methods
      function obj = Conv2DGroupedRFLayer(inputRows, inputCols, nChannels, filterRows, ...
                                             filterCols, nGroups, groupSize, receptiveField, varargin)
         obj = obj@GroupedRFLayer(nGroups, groupSize, receptiveField, nChannels, varargin{:});
         obj.inputRows = inputRows;
         obj.inputCols = inputCols;
         obj.filterRows = filterRows;
         obj.filterCols = filterCols;
         obj.yRows = inputRows - filterRows + 1;
         obj.yCols = inputCols - filterCols + 1;
         
         obj.convGroups = cell(1, obj.nGroups);
         for i = 1:obj.nGroups
            obj.convGroups{i} = Conv2DLayer(inputRows, inputCols, receptiveField, ...
                                               filterRows, filterCols, groupSize, varargin{:});
         end
      end     
            
      function objCopy = copy(obj)
         objCopy = Conv2DGroupedRFLayer(obj.inputRows, obj.inputCols, obj.nChannels, ...
                     obj.filterRows, obj.filterCols, obj.nGroups, obj.groupSize, ...
                     obj.receptiveField);
         
         objCopy.gpuState = obj.gpuState;
         objCopy.connectionTable = obj.connectionTable;
         for i = 1:obj.nGroups
            objCopy.convGroups{i} = obj.convGroups{i}.copy();
         end
      end
   end
   
end

