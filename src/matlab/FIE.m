classdef FIE < handle
    % Full-information estimator for a 2-D planar legged model.
    %
    % State order: [base_position; base_velocity; left_foot; right_foot].
    % The estimator expects measured q, dq, ddq, contact and model-specific
    % kinematics functions supplied as function handles.

    properties (Constant)
        StateSize = 8;
        MeasurementSize = 8;
        ThetaSize = 24;

        LeftFootContact = -1;
        DoubleSupport = 0;
        RightFootContact = 1;
    end

    properties (SetAccess = private)
        dt
        theta
        xFIE = []
        xVector = []
        qHistory = []
        dqHistory = []
        ddqHistory = []
        contactHistory = []
        arrivalState = []
    end

    properties (Dependent)
        x_FIE
        x_vec
    end

    properties (Access = private)
        model
        options
        params

        ADynStack = {}
        bDynStack = {}
        QDynStack = {}
        AMeasStack = {}
        bMeasStack = {}
        QMeasStack = {}

        HArrival = []
        hArrival = []
    end

    methods
        function obj = FIE(dt, theta, model, options)
            if nargin < 3 || isempty(model)
                model = FIE.defaultKinematicsModel();
            end
            if nargin < 4 || isempty(options)
                options = struct();
            end

            obj.dt = dt;
            obj.model = model;
            obj.options = FIE.withDefaultOptions(options);
            obj.setTheta(theta);
        end

        function setData(obj, q, dq, ddq, contact, arrivalState)
            obj.qHistory = obj.asHistory(q, 7, 'q');
            obj.dqHistory = obj.asHistory(dq, 7, 'dq');
            obj.ddqHistory = obj.asHistory(ddq, 7, 'ddq');
            obj.contactHistory = obj.asContactHistory(contact);

            K = size(obj.qHistory, 2);
            if size(obj.dqHistory, 2) ~= K || size(obj.ddqHistory, 2) ~= K
                error('FIE:DataSize', ...
                    'q, dq, and ddq must contain the same number of samples.');
            end
            if numel(obj.contactHistory) ~= K
                error('FIE:ContactSize', ...
                    'contact must contain one value per sample.');
            end

            if nargin < 6 || isempty(arrivalState)
                arrivalState = obj.defaultArrivalState(obj.qHistory(:, 1), obj.dqHistory(:, 1));
            end
            obj.arrivalState = arrivalState(:);
            if numel(obj.arrivalState) ~= obj.StateSize
                error('FIE:ArrivalSize', ...
                    'arrivalState must be an 8-by-1 vector.');
            end
        end

        function value = get.x_FIE(obj)
            value = obj.xFIE;
        end

        function value = get.x_vec(obj)
            value = obj.xVector;
        end

        function appendMeasurement(obj, q, dq, ddq, contact)
            q = obj.asHistory(q, 7, 'q');
            dq = obj.asHistory(dq, 7, 'dq');
            ddq = obj.asHistory(ddq, 7, 'ddq');
            contact = obj.asContactHistory(contact);

            if size(q, 2) ~= 1 || size(dq, 2) ~= 1 || size(ddq, 2) ~= 1 || numel(contact) ~= 1
                error('FIE:AppendSize', ...
                    'appendMeasurement expects one q, dq, ddq, and contact sample.');
            end

            if isempty(obj.qHistory)
                obj.setData(q, dq, ddq, contact);
            else
                obj.qHistory = [obj.qHistory, q];
                obj.dqHistory = [obj.dqHistory, dq];
                obj.ddqHistory = [obj.ddqHistory, ddq];
                obj.contactHistory = [obj.contactHistory, contact];
            end
        end

        function xVector = solve(obj, theta)
            if nargin >= 2 && ~isempty(theta)
                obj.setTheta(theta);
            end
            obj.rebuildConstraints();
            obj.solveOptimization();
            xVector = obj.xVector;
        end

        function xVector = runFIE(obj, theta)
            xVector = obj.solve(theta);
        end

        function xVector = run_FIE(obj, theta)
            xVector = obj.solve(theta);
        end

        function solve_FIE(obj)
            obj.solve();
        end

        function J = finiteDifferenceJacobian(obj, theta, epsilon, useParallel)
            if nargin < 3 || isempty(epsilon)
                epsilon = obj.options.FiniteDifferenceStep;
            end
            if nargin < 4 || isempty(useParallel)
                useParallel = obj.options.UseParallel;
            end

            theta = theta(:);
            base = obj.solve(theta);
            nTheta = numel(theta);
            nOutput = numel(base);
            J = zeros(nOutput, nTheta);

            data = obj.dataStruct();
            qData = data.q;
            dqData = data.dq;
            ddqData = data.ddq;
            contactData = data.contact;
            arrivalData = data.arrivalState;
            dtLocal = obj.dt;
            modelLocal = obj.model;
            optionsLocal = obj.options;

            if useParallel
                parfor j = 1:nTheta
                    thetaPerturbed = theta;
                    thetaPerturbed(j) = thetaPerturbed(j) + epsilon;
                    worker = FIE(dtLocal, thetaPerturbed, modelLocal, optionsLocal);
                    worker.setData(qData, dqData, ddqData, contactData, arrivalData);
                    fPerturbed = worker.solve(thetaPerturbed);
                    J(:, j) = (fPerturbed - base) / epsilon;
                end
            else
                for j = 1:nTheta
                    thetaPerturbed = theta;
                    thetaPerturbed(j) = thetaPerturbed(j) + epsilon;
                    worker = FIE(dtLocal, thetaPerturbed, modelLocal, optionsLocal);
                    worker.setData(qData, dqData, ddqData, contactData, arrivalData);
                    fPerturbed = worker.solve(thetaPerturbed);
                    J(:, j) = (fPerturbed - base) / epsilon;
                end
            end
        end

        function J = jacobian(obj, theta, epsilon)
            J = obj.finiteDifferenceJacobian(theta, epsilon);
        end

        function data = dataStruct(obj)
            data = struct( ...
                'q', obj.qHistory, ...
                'dq', obj.dqHistory, ...
                'ddq', obj.ddqHistory, ...
                'contact', obj.contactHistory, ...
                'arrivalState', obj.arrivalState);
        end

        function update_estimation(obj, ~, q, dq, ddq, contact)
            obj.appendMeasurement(q, dq, ddq, contact);
        end
    end

    methods (Access = private)
        function setTheta(obj, theta)
            theta = theta(:);
            if numel(theta) ~= obj.ThetaSize
                error('FIE:ThetaSize', ...
                    'theta must contain 24 parameters.');
            end

            obj.theta = theta;
            obj.params = struct( ...
                'positionCov', theta(1:3), ...
                'accelCov', theta(4:6), ...
                'stanceFootStd', theta(7:8), ...
                'swingFootStd', theta(9:10), ...
                'jointPositionCov', theta(11:13), ...
                'jointVelocityCov', theta(14:16), ...
                'omegaStd', theta(17), ...
                'initialPositionStd', theta(18:19), ...
                'initialVelocityStd', theta(20:21), ...
                'initialFootStd', theta(22:23), ...
                'linkError', theta(24));
        end

        function rebuildConstraints(obj)
            obj.assertDataReady();
            obj.ADynStack = {};
            obj.bDynStack = {};
            obj.QDynStack = {};
            obj.AMeasStack = {};
            obj.bMeasStack = {};
            obj.QMeasStack = {};

            K = size(obj.qHistory, 2);
            obj.updateArrivalForCurrentTheta();

            first = obj.measurementAt(1);
            obj.appendMeasurementConstraint(first);

            for k = 2:K
                previous = obj.measurementAt(k - 1);
                current = obj.measurementAt(k);
                obj.appendDynamicsConstraint(previous);
                obj.appendMeasurementConstraint(current);
            end
        end

        function solveOptimization(obj)
            D = numel(obj.ADynStack);

            solverOptions = struct();
            solverOptions.ipopt.print_level = obj.options.IpoptPrintLevel;
            solverOptions.ipopt.sb = obj.options.IpoptSilentBanner;
            solverOptions.print_time = obj.options.PrintSolverTime;

            opti = casadi.Opti();
            X = opti.variable(obj.StateSize, D + 1);
            deltaDyn = opti.variable(obj.StateSize, D);
            deltaMeas = opti.variable(obj.MeasurementSize, D + 1);

            objective = X(:, 1)' * obj.HArrival * X(:, 1) + 2 * obj.hArrival' * X(:, 1);
            for k = 1:D
                objective = objective + deltaMeas(:, k)' * obj.QMeasStack{k} * deltaMeas(:, k);
                opti.subject_to(obj.AMeasStack{k} * X(:, k) - obj.bMeasStack{k} + deltaMeas(:, k) == 0);

                objective = objective + deltaDyn(:, k)' * obj.QDynStack{k} * deltaDyn(:, k);
                opti.subject_to(X(:, k + 1) == obj.ADynStack{k} * X(:, k) - obj.bDynStack{k} + deltaDyn(:, k));
            end

            objective = objective + deltaMeas(:, D + 1)' * obj.QMeasStack{D + 1} * deltaMeas(:, D + 1);
            opti.subject_to(obj.AMeasStack{D + 1} * X(:, D + 1) - obj.bMeasStack{D + 1} + deltaMeas(:, D + 1) == 0);

            opti.minimize(objective);
            opti.solver(obj.options.CasadiSolver, solverOptions);

            solution = opti.solve();
            obj.xFIE = solution.value(X);
            obj.xVector = obj.xFIE(:);
        end

        function updateArrivalForCurrentTheta(obj)
            x0 = obj.arrivalState;
            q0 = obj.qHistory(:, 1);
            x0(5:6) = x0(1:2) + obj.leftFootPosition(q0);
            x0(7:8) = x0(1:2) + obj.rightFootPosition(q0);

            CArrival = blkdiag( ...
                diag(obj.params.initialPositionStd .^ 2), ...
                diag(obj.params.initialVelocityStd .^ 2), ...
                diag(obj.params.initialFootStd .^ 2), ...
                diag(obj.params.initialFootStd .^ 2));
            obj.HArrival = obj.informationMatrix(CArrival);
            obj.hArrival = -obj.HArrival * x0;
        end

        function measurement = measurementAt(obj, k)
            q = obj.qHistory(:, k);
            dq = obj.dqHistory(:, k);
            ddq = obj.ddqHistory(:, k);

            measurement = struct();
            measurement.q = q;
            measurement.dq = dq;
            measurement.accel = ddq(1:2);
            measurement.omega = dq(3);
            measurement.rot = FIE.rotation2D(q(3));
            measurement.contact = obj.contactHistory(k);
        end

        function appendDynamicsConstraint(obj, measurement)
            G = zeros(obj.StateSize, obj.StateSize);
            G(1:2, 1:2) = measurement.rot * obj.dt;
            G(1:2, 3:4) = -0.5 * measurement.rot * obj.dt^2;
            G(3:4, 3:4) = -measurement.rot * obj.dt;
            G(5:6, 5:6) = measurement.rot * obj.dt;
            G(7:8, 7:8) = measurement.rot * obj.dt;

            [leftFootProcessCov, rightFootProcessCov] = obj.contactFootCovariances(measurement.contact);

            CInput = blkdiag( ...
                FIE.cov2(obj.params.positionCov), ...
                FIE.cov2(obj.params.accelCov), ...
                leftFootProcessCov, ...
                rightFootProcessCov);

            CDyn = G * CInput * G';
            QDyn = obj.informationMatrix(CDyn);

            ADyn = eye(obj.StateSize);
            ADyn(1:2, 3:4) = obj.dt * eye(2);

            bDyn = zeros(obj.StateSize, 1);
            bDyn(1:2) = -0.5 * measurement.rot * obj.dt^2 * measurement.accel;
            bDyn(3:4) = -obj.dt * measurement.rot * measurement.accel;

            obj.ADynStack{end + 1} = ADyn;
            obj.bDynStack{end + 1} = bDyn;
            obj.QDynStack{end + 1} = QDyn;
        end

        function appendMeasurementConstraint(obj, measurement)
            CJointPosition = FIE.cov2(obj.params.jointPositionCov);
            CJointVelocity = FIE.cov2(obj.params.jointVelocityCov);
            COmega = obj.params.omegaStd^2;

            leftPosition = measurement.rot * obj.leftFootPosition(measurement.q);
            rightPosition = measurement.rot * obj.rightFootPosition(measurement.q);

            JLeft = obj.leftFootJointJacobian(measurement.q);
            JRight = obj.rightFootJointJacobian(measurement.q);
            JLeftBaseJoint = obj.leftFootBaseJointJacobian(measurement.q);
            JRightBaseJoint = obj.rightFootBaseJointJacobian(measurement.q);

            leftVelocity = JLeftBaseJoint * measurement.dq(3:7);
            rightVelocity = JRightBaseJoint * measurement.dq(3:7);

            AMeas = zeros(obj.MeasurementSize, obj.StateSize);
            AMeas(1:2, 1:2) = -eye(2);
            AMeas(1:2, 5:6) = eye(2);
            AMeas(3:4, 1:2) = -eye(2);
            AMeas(3:4, 7:8) = eye(2);
            AMeas(5:6, 3:4) = -eye(2);
            AMeas(7:8, 3:4) = -eye(2);

            bMeas = [leftPosition; rightPosition; leftVelocity; rightVelocity];

            CPositionLeft = measurement.rot * JLeft * CJointPosition * JLeft' * measurement.rot';
            CPositionRight = measurement.rot * JRight * CJointPosition * JRight' * measurement.rot';

            GVelocityLeft = [ ...
                FIE.so2Skew(measurement.omega) * JLeft, ...
                JLeft, ...
                [0 1; -1 0] * obj.leftFootPosition(measurement.q)];
            GVelocityRight = [ ...
                FIE.so2Skew(measurement.omega) * JRight, ...
                JRight, ...
                [0 1; -1 0] * obj.rightFootPosition(measurement.q)];

            CJointVelocityInput = blkdiag(CJointPosition, CJointVelocity, COmega);
            switch measurement.contact
                case obj.RightFootContact
                    CVelocityLeft = diag(obj.params.swingFootStd .^ 2);
                    CVelocityRight = measurement.rot * GVelocityRight * CJointVelocityInput * GVelocityRight' * measurement.rot';
                case obj.LeftFootContact
                    CVelocityLeft = measurement.rot * GVelocityLeft * CJointVelocityInput * GVelocityLeft' * measurement.rot';
                    CVelocityRight = diag(obj.params.swingFootStd .^ 2);
                case obj.DoubleSupport
                    CVelocityLeft = measurement.rot * GVelocityLeft * CJointVelocityInput * GVelocityLeft' * measurement.rot';
                    CVelocityRight = measurement.rot * GVelocityRight * CJointVelocityInput * GVelocityRight' * measurement.rot';
                otherwise
                    error('FIE:Contact', 'Unknown contact value: %g.', measurement.contact);
            end

            CMeas = blkdiag(CPositionLeft, CPositionRight, CVelocityLeft, CVelocityRight);
            QMeas = obj.informationMatrix(CMeas);

            obj.AMeasStack{end + 1} = AMeas;
            obj.bMeasStack{end + 1} = bMeas;
            obj.QMeasStack{end + 1} = QMeas;
        end

        function [leftCov, rightCov] = contactFootCovariances(obj, contact)
            stanceCov = diag(obj.params.stanceFootStd .^ 2);
            swingCov = diag(obj.params.swingFootStd .^ 2);

            switch contact
                case obj.RightFootContact
                    leftCov = swingCov;
                    rightCov = stanceCov;
                case obj.LeftFootContact
                    leftCov = stanceCov;
                    rightCov = swingCov;
                case obj.DoubleSupport
                    leftCov = stanceCov;
                    rightCov = stanceCov;
                otherwise
                    error('FIE:Contact', 'Unknown contact value: %g.', contact);
            end
        end

        function Q = informationMatrix(obj, C)
            C = (C + C') / 2;
            scale = max(1, norm(C, 'fro'));
            C = C + obj.options.CovarianceJitter * scale * eye(size(C));

            [~, cholStatus] = chol(C);
            if cholStatus == 0
                Q = C \ eye(size(C));
            else
                Q = pinv(C);
            end
            Q = (Q + Q') / 2;
        end

        function x0 = defaultArrivalState(obj, q0, dq0)
            x0 = zeros(obj.StateSize, 1);
            x0(1:2) = q0(1:2);
            x0(3:4) = dq0(1:2);
            x0(5:6) = x0(1:2) + obj.leftFootPosition(q0);
            x0(7:8) = x0(1:2) + obj.rightFootPosition(q0);
        end

        function p = leftFootPosition(obj, q)
            p = obj.model.leftFootPosition(q, obj.params.linkError);
            p = p(1:2);
        end

        function p = rightFootPosition(obj, q)
            p = obj.model.rightFootPosition(q, obj.params.linkError);
            p = p(1:2);
        end

        function J = leftFootJointJacobian(obj, q)
            J = obj.model.leftFootJacobian(q, obj.params.linkError);
            J = FIE.selectJacobian(J, [2 2], 1:2, 6:7);
        end

        function J = rightFootJointJacobian(obj, q)
            J = obj.model.rightFootJacobian(q, obj.params.linkError);
            J = FIE.selectJacobian(J, [2 2], 1:2, 4:5);
        end

        function J = leftFootBaseJointJacobian(obj, q)
            J = obj.model.leftFootJacobian(q, obj.params.linkError);
            J = FIE.selectJacobian(J, [2 5], 1:2, 3:7);
        end

        function J = rightFootBaseJointJacobian(obj, q)
            J = obj.model.rightFootJacobian(q, obj.params.linkError);
            J = FIE.selectJacobian(J, [2 5], 1:2, 3:7);
        end

        function assertDataReady(obj)
            if isempty(obj.qHistory)
                error('FIE:MissingData', ...
                    'Call setData before solving the estimator.');
            end
        end
    end

    methods (Static)
        function model = defaultKinematicsModel()
            model = struct();
            model.leftFootPosition = @(q, linkError) pLeftToe_d([0; 0; 0; q(4:7)], linkError);
            model.rightFootPosition = @(q, linkError) pRightToe_d([0; 0; 0; q(4:7)], linkError);
            model.leftFootJacobian = @(q, linkError) J_leftToe_d([0; 0; 0; q(4:7)], linkError);
            model.rightFootJacobian = @(q, linkError) J_rightToe_d([0; 0; 0; q(4:7)], linkError);
        end

        function contact = normalizeContact(contact)
            if isstring(contact) || ischar(contact)
                value = lower(char(contact));
                switch value
                    case {'left', 'leftfootcontact', 'left_foot_contact'}
                        contact = FIE.LeftFootContact;
                    case {'right', 'rightfootcontact', 'right_foot_contact'}
                        contact = FIE.RightFootContact;
                    case {'double', 'doublesupport', 'double_support'}
                        contact = FIE.DoubleSupport;
                    otherwise
                        error('FIE:Contact', 'Unknown contact label: %s.', value);
                end
                return;
            end

            if isobject(contact)
                contact = double(int8(contact));
            else
                contact = double(contact);
            end

            if ~ismember(contact, [-1, 0, 1])
                error('FIE:Contact', 'Contact values must be -1, 0, or 1.');
            end
        end

        function layout = thetaLayout()
            layout.positionCov = 1:3;
            layout.accelCov = 4:6;
            layout.stanceFootStd = 7:8;
            layout.swingFootStd = 9:10;
            layout.jointPositionCov = 11:13;
            layout.jointVelocityCov = 14:16;
            layout.omegaStd = 17;
            layout.initialPositionStd = 18:19;
            layout.initialVelocityStd = 20:21;
            layout.initialFootStd = 22:23;
            layout.linkError = 24;
        end
    end

    methods (Static, Access = private)
        function options = withDefaultOptions(options)
            defaults = struct( ...
                'CasadiSolver', 'ipopt', ...
                'IpoptPrintLevel', 0, ...
                'IpoptSilentBanner', 'yes', ...
                'PrintSolverTime', false, ...
                'CovarianceJitter', 1e-9, ...
                'FiniteDifferenceStep', 1e-8, ...
                'UseParallel', false);

            names = fieldnames(defaults);
            for k = 1:numel(names)
                name = names{k};
                if ~isfield(options, name) || isempty(options.(name))
                    options.(name) = defaults.(name);
                end
            end
        end

        function history = asHistory(value, expectedRows, name)
            if size(value, 1) == expectedRows
                history = value;
            elseif size(value, 2) == expectedRows
                history = value';
            else
                error('FIE:DataSize', ...
                    '%s must be %d-by-K or K-by-%d.', name, expectedRows, expectedRows);
            end
        end

        function contact = asContactHistory(contact)
            contact = contact(:).';
            normalized = zeros(size(contact));
            for k = 1:numel(contact)
                normalized(k) = FIE.normalizeContact(contact(k));
            end
            contact = normalized;
        end

        function C = cov2(entries)
            entries = entries(:);
            C = [entries(1), entries(2); entries(2), entries(3)];
        end

        function R = rotation2D(theta)
            R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
        end

        function S = so2Skew(omega)
            S = omega * [0, -1; 1, 0];
        end

        function J = selectJacobian(J, directSize, rows, columns)
            if isequal(size(J), directSize)
                return;
            end
            J = J(rows, columns);
        end
    end
end
