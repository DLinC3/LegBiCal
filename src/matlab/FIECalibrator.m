classdef FIECalibrator < handle
    % Frank-Wolfe calibration wrapper around FIE.

    properties (SetAccess = private)
        estimator
        groundTruth
        groundTruthVector
        options
    end

    methods
        function obj = FIECalibrator(estimator, groundTruth, options)
            if nargin < 3 || isempty(options)
                options = calibrationOptions();
            end

            obj.estimator = estimator;
            obj.groundTruth = obj.asGroundTruthMatrix(groundTruth);
            obj.groundTruthVector = obj.groundTruth(:);
            obj.options = obj.withDefaultOptions(options);
        end

        function result = run(obj, theta0)
            theta = theta0(:);
            maxIterations = obj.options.MaxIterations;
            nTheta = numel(theta);

            thetaHistory = nan(maxIterations + 1, nTheta);
            lossHistory = nan(maxIterations, 1);
            gradNormHistory = nan(maxIterations, 1);
            expectedDeltaHistory = nan(maxIterations, 1);
            actualDeltaHistory = nan(maxIterations, 1);
            trajectories = cell(min(obj.options.SaveFirstNTrajectories, maxIterations), 1);

            thetaHistory(1, :) = theta.';
            xVector = obj.estimator.solve(theta);
            obj.assertCompatibleGroundTruth(xVector);
            xInitial = obj.estimator.xFIE;
            loss = obj.loss(xVector);

            for iter = 1:maxIterations
                fprintf('Frank-Wolfe iteration %d/%d: loss = %.6g\n', iter, maxIterations, loss);

                lossHistory(iter) = loss;
                sensitivity = obj.estimator.finiteDifferenceJacobian( ...
                    theta, obj.options.FiniteDifferenceStep, obj.options.UseParallel);
                gradient = sensitivity' * (xVector - obj.groundTruthVector);
                gradNormHistory(iter) = norm(gradient);

                vertex = obj.solveLinearMinimizationOracle(gradient);
                direction = vertex - theta;
                alpha = obj.stepSize(iter);
                expectedDeltaHistory(iter) = alpha * (gradient' * direction);

                thetaNext = theta + alpha * direction;
                xNext = obj.estimator.solve(thetaNext);
                lossNext = obj.loss(xNext);

                actualDeltaHistory(iter) = lossNext - loss;
                theta = thetaNext;
                thetaHistory(iter + 1, :) = theta.';

                if iter <= numel(trajectories)
                    trajectories{iter} = obj.estimator.xFIE;
                end

                fprintf('  grad_norm = %.6g, expected_delta = %.6g, actual_delta = %.6g\n', ...
                    gradNormHistory(iter), expectedDeltaHistory(iter), actualDeltaHistory(iter));

                xVector = xNext;
                loss = lossNext;
            end

            result = struct();
            result.thetaFinal = theta;
            result.thetaHistory = thetaHistory;
            result.lossHistory = lossHistory;
            result.gradNormHistory = gradNormHistory;
            result.expectedDeltaHistory = expectedDeltaHistory;
            result.actualDeltaHistory = actualDeltaHistory;
            result.xInitial = xInitial;
            result.xFinal = obj.estimator.xFIE;
            result.groundTruth = obj.groundTruth;
            result.trajectories = trajectories;
            result.options = obj.options;
        end

        function exportResult(~, result, outputDir, time)
            if nargin < 4 || isempty(time)
                time = (0:size(result.groundTruth, 2) - 1).';
            else
                time = time(:);
            end

            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end

            iterations = (1:numel(result.lossHistory)).';
            writetable(table(iterations, result.lossHistory(:), result.gradNormHistory(:), ...
                result.expectedDeltaHistory(:), result.actualDeltaHistory(:), ...
                'VariableNames', {'iter', 'loss', 'grad_norm', 'expected_delta', 'actual_delta'}), ...
                fullfile(outputDir, 'calibration_history.csv'));

            writematrix(result.thetaHistory, fullfile(outputDir, 'theta_history.csv'));

            K = numel(time);
            xFinal = result.xFinal(:, 1:K).';
            xGroundTruth = result.groundTruth(:, 1:K).';
            writematrix([time, xGroundTruth, xFinal], fullfile(outputDir, 'trajectory_ground_truth_final.csv'));
        end

        function plotResult(~, result, time)
            if nargin < 3
                time = [];
            end
            plotFIE(time, result.xFinal, result.groundTruth, result.xInitial);
        end
    end

    methods (Access = private)
        function value = loss(obj, xVector)
            residual = xVector - obj.groundTruthVector;
            value = 0.5 * (residual' * residual);
        end

        function vertex = solveLinearMinimizationOracle(obj, gradient)
            if exist('sdpvar', 'file') ~= 2
                error('FIECalibrator:YALMIPMissing', ...
                    'YALMIP is required for the Frank-Wolfe linear minimization oracle.');
            end

            nTheta = numel(gradient);
            x = sdpvar(nTheta, 1);
            constraints = [obj.options.LowerBound(:) <= x, x <= obj.options.UpperBound(:)];
            epsPSD = obj.options.PsdEpsilon;

            constraints = [constraints, ...
                [x(1), x(2); x(2), x(3)] >= epsPSD * eye(2), ...
                [x(4), x(5); x(5), x(6)] >= epsPSD * eye(2), ...
                [x(11), x(12); x(12), x(13)] >= epsPSD * eye(2), ...
                [x(14), x(15); x(15), x(16)] >= epsPSD * eye(2)];

            settings = sdpsettings('solver', obj.options.LmoSolver, 'verbose', obj.options.LmoVerbose);
            diagnostics = optimize(constraints, gradient(:)' * x, settings);
            if diagnostics.problem ~= 0
                error('FIECalibrator:LMOFailed', ...
                    'Linear minimization oracle failed: %s', diagnostics.info);
            end

            vertex = value(x);
        end

        function alpha = stepSize(obj, iter)
            if isfield(obj.options, 'StepSizeFcn') && ~isempty(obj.options.StepSizeFcn)
                alpha = obj.options.StepSizeFcn(iter);
            else
                alpha = 2 / (obj.options.StepWarmStart + 2 * iter);
            end
        end

        function assertCompatibleGroundTruth(obj, xVector)
            if numel(xVector) ~= numel(obj.groundTruthVector)
                error('FIECalibrator:GroundTruthSize', ...
                    'Ground truth size does not match the FIE trajectory size.');
            end
        end
    end

    methods (Static, Access = private)
        function x = asGroundTruthMatrix(x)
            if size(x, 1) == FIE.StateSize
                return;
            end
            if size(x, 2) == FIE.StateSize
                x = x';
                return;
            end
            error('FIECalibrator:GroundTruthSize', ...
                'groundTruth must be 8-by-K or K-by-8.');
        end

        function options = withDefaultOptions(options)
            defaults = calibrationOptions();
            names = fieldnames(defaults);
            for k = 1:numel(names)
                name = names{k};
                if ~isfield(options, name) || isempty(options.(name))
                    options.(name) = defaults.(name);
                end
            end
        end
    end
end
