classdef BasicMonitor < ProgressMonitor
   % Provides core functionality of ProgressMonitor. Should subclass as needed.
   
   properties
      % number of updates between computing validation error scores
      % (default:100)
      validationInterval     
      validLossFunction % function handle for validation error computation

      % boolean indicating whether or not to compute training error when 
      % validation error is computed (default is true)
      isComputeTrainLoss 
      trainLossFunction % function handle for training error computation
      
      % boolean indicating whether to call report function after losses 
      % are computed (default is true)
      isReport 
      reporter % instance of the Reporter class
      
      % whether or not to store copies of model at each validation point: 
      % {'all', 'best', false} (default is 'best')
      isStoreModels 
   end
   
   properties (SetAccess = protected)
      validLoss = [] % stores history of validation loss values
      trainLoss = [] % stores history of training loss values
      bestUpdate = 0 % update where best validationLoss was achieved
      bestValidLoss = Inf % value of best validationLoss
      nUpdates = 0 % number of updates performed thus far
      
      % cell array of stored models if isStoreModels is 'all', 
      % or only best model if 'best'
      models = {}
   end
   
   methods
      function obj = BasicMonitor(varargin)
         p = inputParser();
         p.addParamValue('validationInterval', 100);
         p.addParamValue('validLossFunction', []);
         p.addParamValue('isComputeTrainLoss', true);
         p.addParamValue('trainLossFunction', []);
         p.addParamValue('isStoreModels', 'best');
         p.addParamValue('isReport', true);
         
         parse(p, varargin{:});
         obj.validationInterval = p.Results.validationInterval;
         obj.validLossFunction = p.Results.validLossFunction;
         obj.isComputeTrainLoss = p.Results.isComputeTrainLoss;
         obj.trainLossFunction = p.Results.trainLossFunction;
         obj.isStoreModels = p.Results.isStoreModels;
         obj.isReport = p.Results.isReport;
         
         % if isReport give a basic ConsoleReporter as default reporter
         if obj.isReport 
            obj.reporter = ConsoleReporter();
         end
      end
      
      function isContinue = update(obj, model, dataManager)
         obj.nUpdates = obj.nUpdates + 1;
         if mod(obj.nUpdates, obj.validationInterval) == 0
            obj.compute_loss_values(model, dataManager);
         end
         
         % useful hook for subclasses to implement stopping criteria
         isContinue = obj.should_continue(); 
      end
      
      function isContinue = should_continue(~)
         % Subclasses can implement own stopping criteria by redefining this
         % function.
         isContinue = true;
      end
      
      function compute_loss_values(obj, model, dataManager)
         % Compute validation loss (if dataManager has validationData)
         if ~isempty(dataManager.validationData)
            isContinue = true;
            tempLoss = 0;
            while isContinue
               [batch, isContinue] = dataManager.validLoss_batch();
               batchSize = size(batch{1}, 2);
               if isempty(obj.validLossFunction)
                  tempLoss = tempLoss + batchSize*model.compute_loss(batch);
               else
                  y = model.output(batch{1});
                  t = batch{end};
                  
                  % have to 'unaverage' losses
                  tempLoss = tempLoss + batchSize*obj.validLossFunction(y, t); 
               end
            end
            
            % 'reaverage' losses
            obj.validLoss = [obj.validLoss, tempLoss/dataManager.validationSize]; 

            if obj.validLoss(end) < obj.bestValidLoss
               obj.bestValidLoss = obj.validLoss(end);
               obj.bestUpdate = obj.nUpdates;
            end
            
            % Store model if appropriate
            switch obj.isStoreModels
               case 'all' % store model snapshot for every validationInterval
                  obj.models = [obj.models, model.copy()];
               case 'best' % only store the best model so far 
                  if obj.bestUpdate == obj.nUpdates
                     obj.models{1} = model.copy();
                  end
            end
         end
         
         % Compute training loss if isComputeTrainLoss
         if obj.isComputeTrainLoss
            isContinue = true;
            tempLoss = 0;
            nExamples = 0;
            while isContinue
               [batch, isContinue] = dataManager.trainLoss_batch();
               batchSize = size(batch{1}, 2);
               if isempty(obj.trainLossFunction)
                  tempLoss = tempLoss + batchSize*model.compute_loss(batch);
               else
                  y = model.output(batch{1});
                  t = batch{end};
                  tempLoss = tempLoss + batchSize*obj.trainLossFunction(y, t);
               end
               nExamples = nExamples + batchSize;
            end
            obj.trainLoss = [obj.trainLoss, tempLoss/nExamples];
         end
         
         % Report results
         if obj.isReport
            obj.reporter.report(obj, model);
         end
      end    
      
      function reset(obj)
         obj.nUpdates = 0;
         obj.models = {};
         obj.bestUpdate = 0;
         obj.bestValidLoss = Inf;
         obj.validLoss = [];
         obj.trainLoss = [];
         
         if ~isempty(obj.reporter)
            obj.reporter.reset();
         end
      end
   end
   
end

