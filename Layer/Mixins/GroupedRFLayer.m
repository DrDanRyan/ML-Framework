classdef GroupedRFLayer < handle
   % A mixin that provides some basic functionality for grouped receptive fields 
   % for the hidden units of a ConvLayer.
   
   properties
      nGroups % G
      groupSize % gS
      receptiveField % rF
      connectionTable % G x C with rF nonzero entries per row
   end
   
   properties (Abstract)
      nChannels
      gpuState
   end
   
   methods
      function obj = GroupedRFLayer(nGroups, groupSize, receptiveField)
         obj.nGroups = nGroups;
         obj.groupSize = groupSize;
         obj.receptiveField = receptiveField;
      end
      
      function init_params(obj)
         % Create random connection table
         obj.connectionTable = obj.gpuState.false(obj.nGroups, obj.nChannels);
         for i = 1:obj.nGroups
            idx = randsample(obj.nChannels, obj.receptiveField)';
            obj.connectionTable(i,idx) = true;
         end
      end

   end  
end

