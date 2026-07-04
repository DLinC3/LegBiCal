classdef estimation_FIE < FIE
    % Deprecated compatibility wrapper.
    % Use FIE for new code.

    methods
        function obj = estimation_FIE(dt, theta, varargin)
            obj@FIE(dt, theta, varargin{:});
        end
    end
end
