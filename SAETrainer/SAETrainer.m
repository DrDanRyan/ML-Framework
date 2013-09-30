classdef SAETrainer < handle
   % A trainer class for training the StackedAutoencoder class.
   % Capable of layerwise pretraining, unsupervised fine-tuning and converting 
   % the resulting deep architecture to a FeedForwardNet.
   
   properties
      layers % a cell array containing instances of the AutoEncoder class
      dataManager
      stepCalculator
      trainingSchedule
   end
   
   methods
      function obj = SAETrainer()
         
      end
      
      function train_layer(obj, layerIdx)
         inputs = obj.get_layer_inputs(layerIdx);
         
      end
      
      function fine_tune(obj)
         
      end
      
      function push_to_GPU(obj)
         
      end
      
      function gather(obj)
         
      end
      
      function reset(obj)
         
      end
   end
   
end

