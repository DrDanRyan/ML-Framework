classdef Trainer < handle
% This is a composite object that manages the training of a model.
%
% Key properties that should be set manually before trian() is called:
%
% dataManager - Stores training and validation data, and serves batches.
%
% stepCalculator - Feeds a batch to the model to collect model statistics
% (for example the gradient) to determine a step direction and size. Some 
% stepCalculator objects require parameters (e.g. learning rate and momentum) 
% in which case parameters are obtained from the parameterSchedule object.
% 
% model - The model that is being trained. The model must have an interface that
% is compatible with the requirements of stepCalculator. Most stepCalculators
% require model to have a method: [grad, output] = gradient(obj, batch); but
% conceivably a stepCalculator may require more information than that from the
% model.
%
% parameterSchedule - updates parameters of stepCalculator as 
% learning progresses. Some stepCalculator objects do not need a
% parameterSchedule (e.g. AdaDelta).
%
% progressMonitor - computes any relevant statistics on training progress
% (on training and/or validation set) and determines when training should 
% terminate. Also responsible for reporting progress statistics to user.
   
   properties
      dataManager % an object that implements the DataManager interface
      stepCalculator % an object that implements the StepCalculator interface
      model % an object that implements the interface required by stepCalculator
      
      % computes and reports performance metrics and can send stop signal 
      % to terminate training
      progressMonitor 
      
      % computes the training parameters used in stepCalculator
      parameterSchedule
   end
   
   methods
      function train(obj, maxUpdates)   
         % Trains model until progressMonitor sends stop signal or maxUpdates is
         % reached.
         isContinue = true;
         nUpdates = 0;
         while isContinue
            nUpdates = nUpdates + 1;
            isContinue = obj.update();
            if nUpdates >= maxUpdates
               break;
            end
         end
      end
      
      
      function isContinue = update(obj)
         % Tells stepCalculator to request batch statistics from model and 
         % update model parameters accordingly. If there is a progressMonitor, 
         % it is told to update as well.
         
         if isempty(obj.dataManager)
            batch = [];
         else
            batch = obj.dataManager.next_batch();
         end
         
         if isempty(obj.parameterSchedule)
            params = [];
         else
            params = obj.parameterSchedule.update();
         end
         
         obj.stepCalculator.take_step(batch, obj.model, params);
         
         if ~isempty(obj.progressMonitor)
            isContinue = obj.progressMonitor.update(obj.model, ...
                                                    obj.dataManager); 
         else
            isContinue = true;
         end
      end
      
      
      function reset(obj)
         % Calls reset() on all properties of the GradientTrainer object
         % that are not empty
         if ~isempty(obj.dataManager)
            obj.dataManager.reset();
         end
         
         if ~isempty(obj.model)
            obj.model.reset();
         end
         
         if ~isempty(obj.progressMonitor)
            obj.progressMonitor.reset();
         end
         
         if ~isempty(obj.parameterSchedule)
            obj.parameterSchedule.reset();
         end
         
         if ~isempty(obj.stepCalculator)
            obj.stepCalculator.reset();
         end
      end
   end      
end

