classdef FixedUpdates < ProgressMonitor
   % Gives stop training signal after a fixed number of updates.
   
   properties
      validationInterval % number of updates between computing validation error scores
      validLossFunction % function handle for validation error computation
      isComputeTrainLoss % whether or not to compute training error when validation error is computed
      trainLossFunction % function handle for training error computation
      isStoreModels % whether or not to store copies of model at each validation point
      models = {} % cell array of stored models if isStoreModels is true
      
      maxUpdates % number of updates to perform in total before terminating
      nUpdates % number of updates performed thus far
   end
   
   methods
      function obj = FixedUpdates(maxUpdates, varargin)
         obj.maxUpdates = maxUpdates;
         obj.nUpdates = 0;
         
         function validationLoss = default_validLossFunction(model, dataManager)
            validationLoss = [];
            if ~isempty(dataManager.validationData)
               validationLoss = model.compute_loss(dataManager.validationData);
            end
         end
         
         function trainLoss = default_trainLossFunction(model, dataManager)
            trainLoss = model.compute_loss(dataManager.trainingData);
         end
         
         p = inputParser();
         p.addParamValue('validationInterval', maxUpdates);
         p.addParamValue('validLossFunction', @default_validLossFunction);
         p.addParamValue('isComputeTrainLoss', true);
         p.addParamValue('trainLossFunction', @default_trainLossFunction);
         p.addParamValue('isStoreModels', false);
         
         parse(p, varargin{:});
         obj.validationInterval = p.Results.validationInterval;
         obj.validLossFunction = p.Results.validLossFunction;
         obj.isComputeTrainLoss = p.Results.isComputeTrainLoss;
         obj.trainLossFunction = p.Results.trainLossFunction;
         obj.isStoreModels = p.Results.isStoreModels;
      end
      
      function isContinue = update(obj, model, dataManager)
         obj.nUpdates = obj.nUpdates + 1;
         isContinue = obj.nUpdates < obj.maxUpdates;
         
         if mod(obj.nUpdates, obj.validationInterval)
            if obj.isStoreModels
               obj.models = [obj.models, model.copy()];
            end
            
            validLoss = obj.validLossFunction(model, dataManager);
            
            if obj.isComputeTrainLoss
               trainLoss = obj.trainLossFunction(model, dataManager);
               fprintf('\nupdate %d:  train: %.4f   valid: %.4f', obj.nUpdates, ...
                           trainLoss, validLoss);
            else
               fprintf('\nupdate %d:  valid: %.4f', obj.nUpdates, validLoss);
            end
         end
      end
      
      function reset(obj)
         obj.nUpdates = 0;
         obj.models = {};
      end
   end
   
end

