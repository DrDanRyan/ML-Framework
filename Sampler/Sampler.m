classdef Sampler < handle
   % This defines the Sampler interface
   
   methods (Abstract)
      [sample, out_of_sample] = sample(obj, data)
   end
   
end

