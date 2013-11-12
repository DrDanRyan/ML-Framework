classdef BasicMonitor < ProgressMonitor
   % Provides core functionality of ProgressMonitor. Should subclass and 
   
   properties
      validationInterval % number of updates between computing validation error scores
      validLossFunction % function handle for validation error computation
      isComputeTrainLoss % whether or not to compute training error when validation error is computed
      trainLossFunction % function handle for training error computation
      validLoss = [] 
      trainLoss = []
      
      isReport % boolean indicating whether to call report function after losses are computed
      
      isStoreModels % whether or not to store copies of model at each validation point: {'all', 'best', false}
      models = {} % cell array of stored models if isStoreModels is 'all', or only best model if 'best'
      
      bestUpdate = 0 % update where best validationLoss was achieved
      bestValidLoss = Inf % value of best validationLoss
      nUpdates = 0 % number of updates performed thus far
   end
   
   methods
      function obj = BasicMonitor(varargin)
         
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
         p.addParamValue('validationInterval', 100);
         p.addParamValue('validLossFunction', @default_validLossFunction);
         p.addParamValue('isComputeTrainLoss', true);
         p.addParamValue('trainLossFunction', @default_trainLossFunction);
         p.addParamValue('isStoreModels', 'best');
         p.addParamValue('isReport', true);
         
         parse(p, varargin{:});
         obj.validationInterval = p.Results.validationInterval;
         obj.validLossFunction = p.Results.validLossFunction;
         obj.isComputeTrainLoss = p.Results.isComputeTrainLoss;
         obj.trainLossFunction = p.Results.trainLossFunction;
         obj.isStoreModels = p.Results.isStoreModels;
         obj.isReport = p.Results.isReport;
      end
      
      function isContinue = update(obj, model, dataManager)
         obj.nUpdates = obj.nUpdates + 1;
         if mod(obj.nUpdates, obj.validationInterval) == 0
            obj.compute_loss_values(model, dataManager);
         end
         isContinue = obj.should_continue();
      end
      
      function isContinue = should_continue(~)
         isContinue = true;
      end
      
      function compute_loss_values(obj, model, dataManager)
         obj.validLoss = [obj.validLoss, obj.validLossFunction(model, dataManager)];
         if obj.validLoss(end) < obj.bestValidLoss
            obj.bestValidLoss = obj.validLoss(end);
            obj.bestUpdate = obj.nUpdates;
         end
            
         % Store model if appropriate
         switch obj.isStoreModels
            case 'all' % store all model snapshots for each validationInterval
               obj.models = [obj.models, model.copy()];
            case 'best' % only store the best model so far (overwrite previous best)
               if obj.bestUpdate == obj.nUpdates
                  obj.models{1} = model.copy();
               end
         end

         if obj.isComputeTrainLoss
            obj.trainLoss = [obj.trainLoss, obj.trainLossFunction(model, dataManager)];
         end
         
         if obj.isReport
            obj.report();
         end
      end
      
      function report(obj)
         if isempty(obj.trainLoss)
            fprintf('\nupdate %d:  valid: %.4f', obj.nUpdates, obj.validLoss(end));
         else
            fprintf('\nupdate %d:  train: %.4f  valid: %.4f', obj.nUpdates, ...
                     obj.trainLoss(end), obj.validLoss(end));
         end
      end
      
      function plot_training_curves(obj)
         x = obj.validationInterval:obj.validationInterval:obj.nUpdates;
         figure()
         plot(x, obj.validLoss, 'b')
         hold on
         YLim = get(gca, 'YLim');
         plot([obj.bestUpdate, obj.bestUpdate], YLim, 'g-')
         if ~isempty(obj.trainLoss)
            plot(x, obj.trainLoss, 'r')
         end
         hold off
      end      
      
      function reset(obj)
         obj.nUpdates = 0;
         obj.models = {};
         obj.bestUpdate = 0;
         obj.bestValidLoss = Inf;
         obj.validLoss = [];
         obj.trainLoss = [];
      end
   end
   
end

