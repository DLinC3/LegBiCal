function result = main(inputData, options)
%MAIN Compatibility entry point for the Matlab calibration prototype.
%
% Provide a MAT file or struct containing either:
%   1. log.flow, log.estimate, and log.groundtruth, or
%   2. data and xGroundTruth.

if nargin < 1
    error('main:MissingInput', ...
        'Call main(inputData). See runCalibration.m for the data format.');
end

if nargin < 2
    options = calibrationOptions();
end

result = runCalibration(inputData, options);
end
