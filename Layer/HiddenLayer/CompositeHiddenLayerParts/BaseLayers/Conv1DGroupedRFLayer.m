classdef Conv1DGroupedRFLayer < GroupedRFLayer
   
   properties
      inputSize
      filterSize
      ySize
   end
   
   methods
      function obj = Conv1DGroupedRFLayer(inputSize, nChannels, filterSize, ...
                                             nGroups, groupSize, receptiveField, varargin)
         obj = obj@GroupedRFLayer(nGroups, groupSize, receptiveField, nChannels, varargin{:});
         obj.inputSize = inputSize;
         obj.filterSize = filterSize;
         obj.ySize = inputSize - filterSize + 1;
         
         obj.convGroups = cell(1, obj.nGroups);
         for i = 1:obj.nGroups
            obj.convGroups{i} = Conv1DLayer(inputSize, receptiveField, ...
                                               filterSize, groupSize, varargin{:});
         end
      end     
            
      function objCopy = copy(obj)
         objCopy = Conv1DGroupedRFLayer(obj.inputSize, obj.nChannels, ...
                     obj.filterSize, obj.nGroups, obj.groupSize, obj.receptiveField);
         
         objCopy.gpuState = obj.gpuState;
         objCopy.connectionTable = obj.connectionTable;
         for i = 1:obj.nGroups
            objCopy.convGroups{i} = obj.convGroups{i}.copy();
         end
      end
      
   end
end

