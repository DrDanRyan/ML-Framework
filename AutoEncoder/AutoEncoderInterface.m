classdef AutoEncoderInterface < SupervisedModel
   % Autoencoders must have same interface as SupervisedModel and the
   % additional method ``encode'' which is useful for layerwise pretraining
   
   methods (Abstract)
      h = encode(obj, x)
   end
   
end

