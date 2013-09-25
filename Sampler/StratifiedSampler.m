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
      
      function sampleIdx = sample(obj, targets)
         dataSize = size(targets, 2);
         fullIdx = 1:dataSize;
         
         positives = fullIdx(targets==1);
         posSize = length(positives);
         
         negatives = fullIdx(targets ~= 1);
         negSize = length(negatives);
         
         posSample = positives(randsample(posSize, round(obj.proportion*posSize)));
         negSample = negatives(randsample(negSize, round(obj.proportion*negSize)));
         sampleIdx = [posSample, negSample];
         sampleIdx = sampleIdx(randperm(length(sampleIdx)));
      end
   end
   
end

