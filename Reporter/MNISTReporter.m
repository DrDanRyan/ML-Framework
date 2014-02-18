classdef MNISTReporter < Reporter
   % A reporter class for visualizing MNIST filters (not convolutional
   % filters) during training
   
   properties
      consoleReporter
      rows
      cols
   end
   
   methods
      function obj = MNISTReporter(rows, cols)
         obj.consoleReporter = ConsoleReporter();
         obj.rows = rows;
         obj.cols = cols;
         obj.init_figure();
      end
      
      function init_figure(obj)
         for i = 1:obj.rows
            for j = 1:obj.cols
               subtightplot(obj.rows, obj.cols, obj.cols*(i-1) + j, ...
                              [.01, .01]);
               image(zeros(28, 28), 'CDataMapping', 'scaled');
               colormap('gray');
               axis off tight
            end
         end
         drawnow;
      end
      
      function report(obj, progressMonitor, model)
         if isa(model, 'AutoEncoder')
            W = model.encodeLayer.params{1};
         else % Assume FeedForwardNet
            W = model.hiddenLayers{1}.params{1};
         end
         W = bsxfun(@rdivide, W, sqrt(sum(W.*W, 2))); % normalize filters
         
         for i = 1:obj.rows
            for j = 1:obj.cols
               idx = obj.cols*(i-1) + j;
               subtightplot(obj.rows, obj.cols, idx, [.01, .01]);
               image(reshape(W(idx,:), 28, 28)', 'CDataMapping', 'scaled');
               maxAbs = gather(max(abs(W(idx,:))));
               set(gca, 'CLim', [-maxAbs, maxAbs]);
               colormap('gray');
               axis off tight
            end
         end
         drawnow;
         obj.consoleReporter.report(progressMonitor, model);
      end
      
      function reset(obj)
         close;
         obj.init_figure();
      end
      
   end
end

