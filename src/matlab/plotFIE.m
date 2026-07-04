function plotFIE(t, xFIE, xGroundTruth, xInitial)
%PLOTFIE Plot planar FIE trajectory against ground truth.

if nargin == 1 && isstruct(t)
    input = t;
    if isfield(input, 'estimate')
        xFIE = input.estimate.x_FIE;
        xGroundTruth = input.groundtruth.x;
        t = input.estimate.t;
    elseif isfield(input, 'xFinal')
        xFIE = input.xFinal;
        xGroundTruth = input.groundTruth;
        xInitial = input.xInitial;
        t = [];
    else
        error('plotFIE:Input', 'Unsupported struct input.');
    end
end

if nargin < 4
    xInitial = [];
end

xFIE = asStateMatrix(xFIE);
xGroundTruth = asStateMatrix(xGroundTruth);
K = size(xFIE, 2);
if isempty(t)
    t = (0:K - 1).';
else
    t = t(:);
end

figure(1); clf
labels = {'x (m)', 'z (m)', 'v_x (m/s)', 'v_z (m/s)'};
for idx = 1:4
    subplot(2, 2, idx)
    plot(t, xFIE(idx, 1:K), 'LineWidth', 1.2); hold on
    plot(t, xGroundTruth(idx, 1:K), '--', 'LineWidth', 1.0);
    if ~isempty(xInitial)
        xInitial = asStateMatrix(xInitial);
        plot(t, xInitial(idx, 1:K), ':', 'LineWidth', 1.0);
        legend('FIE final', 'GT', 'FIE initial');
    else
        legend('FIE', 'GT');
    end
    ylabel(labels{idx});
    grid on
end

figure(2); clf
errorLabels = {'x error', 'z error', 'v_x error', 'v_z error'};
for idx = 1:4
    subplot(2, 2, idx)
    plot(t, abs(xFIE(idx, 1:K) - xGroundTruth(idx, 1:K)), 'LineWidth', 1.0);
    ylabel(errorLabels{idx});
    grid on
end

figure(3); clf
footLabels = {'left foot x', 'left foot z', 'right foot x', 'right foot z'};
for idx = 1:4
    stateIdx = idx + 4;
    subplot(2, 2, idx)
    plot(t, xFIE(stateIdx, 1:K), 'LineWidth', 1.2); hold on
    plot(t, xGroundTruth(stateIdx, 1:K), '--', 'LineWidth', 1.0);
    legend('FIE', 'GT');
    ylabel(footLabels{idx});
    grid on
end
end

function x = asStateMatrix(x)
if size(x, 1) == FIE.StateSize
    return;
end
if size(x, 2) == FIE.StateSize
    x = x';
    return;
end
error('plotFIE:StateSize', 'State trajectory must be 8-by-K or K-by-8.');
end
