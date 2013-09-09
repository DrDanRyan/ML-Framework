classdef ProportionSubsampler < Sampler
   
   properties
      proportion
   end
   
   methods
      function obj = ProportionSubsampler(proportion)
         obj.proportion = proportion;
      end
      
      function sample = sample(obj, data)
         dataSize = size(data, 2);
         sampleSize = round(obj.proportion*dataSize);
         sample = data(:, randsample(dataSize, sampleSize));
      end
   end
   
end

