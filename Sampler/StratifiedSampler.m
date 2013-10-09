classdef StratifiedSampler < Sampler
   % Feed in binary targets, returns an index vector for a sample with the
   % same ratio of positive to negative examples present in the original
   % data. The sampler samples the positive and negative examples
   % separately and then shuffles together the result.
   
   properties
      proportion
   end
   
   methods
      function obj = StratifiedSampler(proportion)
         obj.proportion = proportion;
      end
      
      function [sampleIdx, out_of_sample] = sample(obj, idxs, targets)       
         positives = idxs(targets==1);
         posSize = length(positives);
         
         negatives = idxs(targets ~= 1);
         negSize = length(negatives);
         
         posSample = positives(randsample(posSize, round(obj.proportion*posSize)));
         negSample = negatives(randsample(negSize, round(obj.proportion*negSize)));
         sampleIdx = [posSample, negSample];
         sampleIdx = sampleIdx(randperm(length(sampleIdx)));
         out_of_sample = setdiff(idxs, sampleIdx);
      end
   end
   
end

