classdef ProportionSubsampler < Sampler
   
   properties
      proportion
   end
   
   methods
      function obj = ProportionSubsampler(proportion)
         obj.proportion = proportion;
      end
      
      function [sample, out_of_sample] = sample(obj, data)
         dataSize = size(data, 2);
         sampleSize = round(obj.proportion*dataSize);
         
         sampleIdx = randsample(dataSize, sampleSize);
         out_of_sampleIdx = setdiff(1:dataSize, sampleIdx);
         
         sample = data(:, sampleIdx);
         out_of_sample = data(:, out_of_sampleIdx);
      end
   end
   
end

