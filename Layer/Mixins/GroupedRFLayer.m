classdef GroupedRFLayer < handle
   % A mixin that provides some basic functionality for grouped receptive fields 
   % for the hidden units of a ConvLayer.
   
   properties
      nGroups   % (G) number of separate filter groups
      groupSize % (gS) number of filters in each group
      receptiveField % (rF) number of inputs channels that each group connects to
      nChannels % (C) number of input channels
      nFilters  % (nF = gS*nG)
      
      connectionTable % a G x C logical matrix where each row has rF true values
      convGroups % a cell array of length nGroups full of Conv1DLayer or Conv2DLayer objects
      gpuState
   end   
   
   methods
      function obj = GroupedRFLayer(nGroups, groupSize, receptiveField, nChannels, varargin)
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('gpu', []);
         parse(p, varargin{:});
         obj.gpuState = GPUState(p.Results.gpu);
         
         obj.nGroups = nGroups;
         obj.groupSize = groupSize;
         obj.receptiveField = receptiveField;
         obj.nChannels = nChannels;
         obj.nFilters = groupSize*nGroups;
         
         obj.connectionTable = obj.gpuState.false(obj.nGroups, obj.nChannels);
         for i = 1:obj.nGroups
            obj.connectionTable(i,randsample(nChannels, receptiveField)) = true;
         end
      end
      
      function y = feed_forward(obj, x, ~)
         N = size(x, 2);
         if ndims(x) == 3 % 1D convGroups
            y = obj.gpuState.nan(obj.nFilters, N, obj.ySize);
         elseif ndims(x) == 4 % 2D convGroups
            y = obj.gpuState.nan(obj.nFilters, N, obj.yRows, obj.yCols);
         end
         startIdx = 1;
         for i = 1:obj.nGroups
            stopIdx = startIdx + obj.groupSize - 1;
            y(startIdx:stopIdx,:,:,:) = ...
               obj.convGroups{i}.feed_forward(x(obj.connectionTable(i,:),:,:,:));
            startIdx = stopIdx + 1;
         end
      end
      
      function [grad, dLdx] = backprop(obj, x, dLdy)
         dLdx = obj.gpuState.zeros(size(x));
         grad = cell(1, 2*obj.nGroups);
         startIdx = 1;
         for i = 1:obj.nGroups
            stopIdx = startIdx + obj.groupSize - 1;
            connections = obj.connectionTable(i,:);
            [grad(2*(i-1)+1:2*i), dLdxGroup] = ...
               obj.convGroups{i}.backprop(x(connections,:,:,:), dLdy(startIdx:stopIdx,:,:,:));
            dLdx(connections,:,:,:) = dLdx(connections,:,:,:) + dLdxGroup;
            startIdx = stopIdx + 1;
         end
      end
      
      function increment_params(obj, delta)
         for i = 1:obj.nGroups
            obj.convGroups{i}.increment_params(delta(2*(i-1)+1:2*i));
         end
      end
      
      function push_to_GPU(obj)
         for i = 1:obj.nGroups
            obj.convGroups{i}.push_to_GPU();
         end
         obj.gpuState.isGPU = true;
      end
      
      function gather(obj)
         for i = 1:obj.nGroups
            obj.convGroups{i}.gather();
         end
         obj.gpuState.isGPU = false;
      end
      
      function init_params(obj)
         obj.connectionTable = obj.gpuState.false(obj.nGroups, obj.nChannels);
         for i = 1:obj.nGroups
            obj.convGroups{i}.init_params();
            obj.connectionTable(i,randsample(obj.nChannels, obj.receptiveField)) = true;            
         end
      end
      
   end
end

