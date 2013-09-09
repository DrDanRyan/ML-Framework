classdef Sampler < handle
   % This defines the Sampler interface
   
   methods (Abstract)
      sample = sample(obj, data)
   end
   
end

