classdef FixedUpdates < ProgressMonitor
   % Gives stop training signal after a fixed number of updates.
   
   properties
      validationInterval % number of updates between computing validation error scores
      validLossFunction % function handle for validation error computation
      isComputeTrainLoss % whether or not to compute training error when validation error is computed
      trainLossFunction % function handle for training error computation
      validLoss = [] 
      trainLoss = []
      isPlotTrainingCurves % whether or not to plot the trainLoss and validLoss training curves
                           % after training terminates
      
      isStoreModels % whether or not to store copies of model at each validation point
      models = {} % cell array of stored models if isStoreModels is 'all', or only best model if 'best'
      
      bestUpdate = 0 % update where best validationLoss was achieved
      bestValidLoss = Inf % value of best validationLoss
      
      maxUpdates % number of updates to perform in total before terminating
      nUpdates = 0 % number of updates performed thus far
   end
   
   methods
      function obj = FixedUpdates(maxUpdates, varargin)
         obj.maxUpdates = maxUpdates;
         
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
         p.addParamValue('isStoreModels', 'best');
         p.addParamValue('isPlotTrainingCurves', true);
         
         parse(p, varargin{:});
         obj.validationInterval = p.Results.validationInterval;
         obj.validLossFunction = p.Results.validLossFunction;
         obj.isComputeTrainLoss = p.Results.isComputeTrainLoss;
         obj.trainLossFunction = p.Results.trainLossFunction;
         obj.isStoreModels = p.Results.isStoreModels;
         obj.isPlotTrainingCurves = p.Results.isPlotTrainingCurves;
      end
      
      function isContinue = update(obj, model, dataManager)
         obj.nUpdates = obj.nUpdates + 1;
         if mod(obj.nUpdates, obj.validationInterval) == 0
            obj.compute_loss_values(model, dataManager);
         end
         isContinue = obj.nUpdates < obj.maxUpdates;
         if ~isContinue && obj.isPlotTrainingCurves
            obj.plot_training_curves()
         end
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
            fprintf('\nupdate %d:  train: %.4f   valid: %.4f', obj.nUpdates, ...
                        obj.trainLoss(end), obj.validLoss(end));
         else
            fprintf('\nupdate %d:  valid: %.4f', obj.nUpdates, obj.validLoss(end));
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

