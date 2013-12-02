classdef Conv1DHiddenLayer < matlab.mixin.Copyable
   % A convolution hidden layer for multiple channels of 1D signals with
   % max pooling.
   
   properties
      poolSize % (pS) number of units to maxpool over
      nFilters % (nF) number of convolution filters
      inputSize % (X) length of each 1D inputs signal
      nChannels % (C) number of input channels
      filterSize % (fS) length of the filter on each channel
      
      initScale % used for filter initialization
      outputSize % not specified by user, derived from other params at contruction
   end
   
   methods
      function obj = Conv1DHiddenLayer(inputSize, nChannels, filterSize, nFilters, poolSize, varargin)
         obj.inputSize = inputSize;
         obj.nChannels = nChannels;
         obj.filterSize = filterSize;
         obj.nFilters = nFilters;
         obj.poolSize = poolSize;
         obj.outputSize = floor((inputSize - filterSize + 1)/poolSize);
         
         p = inputParser();
         p.addParamValue('initScale', .005);
         parse(p, varargin{:});
         obj.initScale = p.Results.initScale;
      end
   end
   
end

