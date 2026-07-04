function result = runCalibration(inputData, options)
%RUNCALIBRATION Run FIE calibration from an existing data/log file.
%
% inputData can be a MAT file path, a struct with fields data/xGroundTruth,
% or an old-style log struct with flow, estimate, and groundtruth fields.

if nargin < 2 || isempty(options)
    options = calibrationOptions();
else
    options = mergeOptions(calibrationOptions(), options);
end
options.Estimator.FiniteDifferenceStep = options.FiniteDifferenceStep;
options.Estimator.UseParallel = options.UseParallel;

[data, xGroundTruth] = parseInputData(inputData);

model = FIE.defaultKinematicsModel();
estimator = FIE(data.dt, options.Theta0, model, options.Estimator);

arrivalState = [];
if isfield(data, 'arrivalState')
    arrivalState = data.arrivalState;
end
estimator.setData(data.q, data.dq, data.ddq, data.contact, arrivalState);

calibrator = FIECalibrator(estimator, xGroundTruth, options);
result = calibrator.run(options.Theta0);

if isfield(options, 'OutputDirectory') && ~isempty(options.OutputDirectory)
    time = [];
    if isfield(data, 't')
        time = data.t;
    end
    calibrator.exportResult(result, options.OutputDirectory, time);
end
end

function [data, xGroundTruth] = parseInputData(inputData)
if ischar(inputData) || isstring(inputData)
    loaded = load(inputData);
else
    loaded = inputData;
end

if isfield(loaded, 'log')
    [data, xGroundTruth] = dataFromLog(loaded.log);
    return;
end

if isfield(loaded, 'flow') && isfield(loaded, 'groundtruth')
    [data, xGroundTruth] = dataFromLog(loaded);
    return;
end

if isfield(loaded, 'data')
    data = loaded.data;
elseif isfield(loaded, 'calibrationData')
    data = loaded.calibrationData;
else
    error('runCalibration:MissingData', ...
        'Input must contain data, calibrationData, or log.');
end

if isfield(loaded, 'xGroundTruth')
    xGroundTruth = loaded.xGroundTruth;
elseif isfield(loaded, 'xGT')
    xGroundTruth = loaded.xGT;
elseif isfield(data, 'xGroundTruth')
    xGroundTruth = data.xGroundTruth;
else
    error('runCalibration:MissingGroundTruth', ...
        'Input must contain xGroundTruth or xGT.');
end

required = {'q', 'dq', 'ddq', 'contact', 'dt'};
for k = 1:numel(required)
    if ~isfield(data, required{k})
        error('runCalibration:MissingField', ...
            'data.%s is required.', required{k});
    end
end
end

function merged = mergeOptions(defaults, overrides)
merged = defaults;
names = fieldnames(overrides);
for k = 1:numel(names)
    name = names{k};
    if isstruct(overrides.(name)) && isfield(merged, name) && isstruct(merged.(name))
        merged.(name) = mergeOptions(merged.(name), overrides.(name));
    else
        merged.(name) = overrides.(name);
    end
end
end

function [data, xGroundTruth] = dataFromLog(logData)
data = struct();
data.q = logData.flow.q;
data.dq = logData.flow.dq;
data.ddq = logData.flow.ddq;
data.contact = logData.estimate.contact;
data.t = logData.estimate.t(:);
data.dt = median(diff(data.t));
xGroundTruth = logData.groundtruth.x;
end
