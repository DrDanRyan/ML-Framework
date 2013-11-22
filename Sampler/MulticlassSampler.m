classdef MulticlassSampler < Sampler
   % A stratified sampler for multiclass targets
   
   properties
      proportion
   end
   
   methods
      function obj = MulticlassSampler(proportion)
         obj.proportion = proportion;
      end
      
      function [sampleIdx, out_of_sample] = sample(obj, idxs, targets)       
         D = size(targets, 1);
         nSamples = round(obj.proportion*sum(targets, 2));
         sampleIdx = [];
         for i = 1:D
            candidates = idxs(targets(i,:) == 1);
            sampleIdx = [sampleIdx, randsample(candidates, nSamples(i))]; %#ok<AGROW>
         end
         
         sampleIdx = randsample(sampleIdx, size(sampleIdx, 2));
         out_of_sample = setdiff(idxs, sampleIdx);
      end
   end   
end

