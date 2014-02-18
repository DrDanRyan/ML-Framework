classdef ImputingDataManager < DataManager
   
   properties
      isNaN       % logical array same size as trainingData indicating
                  % where trainingData had NaN values originally
      oldBatchIdx % batchIdx used to generate last mini-batch
   end
   
   methods
      function obj = ImputingDataManager(trainingData, validationData, varargin)
         obj = obj@DataManager(trainingData, validationData, varargin{:});
         obj.isNaN = isnan(obj.trainingData{1});
         obj.trainingData{1}(obj.isNaN) = 0;
      end
      
      function batch = next_batch(obj)
         if isempty(obj.batchSize) % full batch
            batch = next_batch@DataManager(obj);
            batch = [batch, obj.isNaN];
         else % mini-batch
            obj.oldBatchIdx = obj.batchIdx;
            batch = next_batch@DataManager(obj);
            batch = [batch, obj.isNaN(:,obj.oldBatchIdx:obj.stopIdx,:,:)];
         end        
      end
      
      function update_imputed_data(obj, xNew)
         if isempty(obj.batchSize) % full batch
            obj.trainingData{1}(obj.isNaN) = xNew;
         else % mini-batch
            obj.trainingData{1}(...
               obj.isNaN(:,obj.oldBatchIdx:obj.stopIdx,:,:)) = xNew;
         end
      end
      
      function shuffle_training_data(obj)
         permvec = randperm(obj.trainingSize);
         obj.trainingData = cellfun(@(v) v(:,permvec,:,:), obj.trainingData, ...
                                                      'UniformOutput', false);
         obj.isNaN = obj.isNaN(:,permvec,:,:);
         obj.batchIdx = 1;
      end
      
      function reset(obj)
         reset@DataManager(obj);
         obj.trainingData{1}(obj.isNaN) = 0;
      end
      
   end
end