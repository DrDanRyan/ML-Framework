classdef ReuseValsLayer < handle
   % A mixin that controls memory vs. computation trade-off for storing
   % feed-forward values that can be reused during backprop
   
   properties
      isReuseVals
   end
   
   methods
      function obj = ReuseValsLayer(varargin)
         p = inputParser();
         p.KeepUnmatched = true;
         p.addParamValue('reuseVals', true);
         parse(p, varargin{:});
         obj.isReuseVals = p.Results.reuseVals;
      end
   end
   
end

