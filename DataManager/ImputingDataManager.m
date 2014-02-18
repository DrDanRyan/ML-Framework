classdef ImputingDataManager < DataManager
   
   properties
      isNaN       % logical array same size as trainingData indicating
                  % where trainingData had NaN values originally
      oldBatchIdx % batchIdx used to generate last mini-batch
      stopIdx  % stopIdx used to generate last mini-batch
   end
   
   methods
      function obj = ImputingDataManager(trainingData, validationData, varargin)
         obj = obj@DataManager(trainingData, validationData, varargin{:});
         obj.isNaN = isnan(obj.trainingData{1});
         obj.trainingData{1}(obj.isNaN) = 0;
      end
      
      function batch = next_batch(obj)
         if isempty(obj.batchSize) % full batch
            batch = obj.trainingData;
            batch{end+1} = obj.isNaN;
         else % mini-batch
            obj.stopIdx = min(obj.trainingSize, obj.batchIdx + obj.batchSize - 1);
            batch = cellfun(@(v) v(:,obj.batchIdx:obj.stopIdx,:,:), obj.trainingData, ...
                              'UniformOutput', false);
            batch{end+1} = obj.isNaN(:,obj.batchIdx:obj.stopIdx,:,:);
            
            obj.oldBatchIdx = obj.batchIdx;
            if obj.stopIdx == obj.trainingSize
               obj.shuffle_training_data();
            else
               obj.batchIdx = obj.stopIdx + 1;
            end
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
         if ~isempty(obj.isNaN) % need to avoid this line during DataManager initialize
            obj.isNaN = obj.isNaN(:,permvec,:,:);
         end
         obj.batchIdx = 1;
      end
      
      function reset(obj)
         reset@DataManager(obj);
         obj.trainingData{1}(obj.isNaN) = 0;
      end
      
   end
end