///////////////////////////////////////////////////////////////////////////////
// g1cal motion-only overlay.
//
// MotionFIEProblem mirrors the pinned ContactAnitescuIDProblem builder with the
// identification machinery removed:
//   - ordinary StateMultibody (nx=71, ndx=70), no identified-joint requirement;
//   - running knots: legacy weight mapping (w^2 activation, cost coefficient
//     10, Euler dt wrapping) minus the parameter blocks;
//   - arrival: ActionModelMotionArrival with a 70-dim control cost whose
//     weights are zero in legacy mode (official arrival has no motion P0).
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <vector>

#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/contact_id/actuations/floating-base-estimation.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"

#include "g1cal/motion_arrival_action.hpp"
#include "g1cal/covariance_precision.hpp"
#include "g1cal/motion_fwddyn_action.hpp"

namespace g1cal
{

class MotionFIEProblem
{
public:
    typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
    typedef crocoddyl::CostModelSumTpl<double> CostModelSum;
    typedef crocoddyl::ActionModelAbstractTpl<double> ActionModelAbstract;

    MotionFIEProblem(const pinocchio::Model &rmodel,
                     const std::vector<std::string> &contact_frame_names,
                     const crocoddyl::ContactIDWeights &weights,
                     const DiagonalCovariancePrecision &covariance =
                         DiagonalCovariancePrecision(),
                     const int newton_max_iters = 300)
        : rmodel_(rmodel),
          contact_frame_names_(contact_frame_names),
          state_(boost::make_shared<StateMultibody>(
              boost::make_shared<pinocchio::Model>(rmodel_))),
          actuation_(boost::make_shared<
                     crocoddyl::ActuationModelFloatingBaseEstimationTpl<double>>(
              state_)),
          weights_(weights),
          covariance_(covariance),
          newton_max_iters_(newton_max_iters)
    {
        if (newton_max_iters_ <= 0)
            throw std::runtime_error("newton_max_iters must be positive");
        if (static_cast<int>(state_->get_nx()) != rmodel_.nq + rmodel_.nv)
        {
            throw std::runtime_error("motion state nx != nq+nv");
        }
        covariance_.validate(state_->get_ndx(), actuation_->get_nu());
    }

    boost::shared_ptr<crocoddyl::ShootingProblem> createEstimationProblem(
        const Eigen::VectorXd &x0, const double timestep,
        const std::size_t n_knots,
        const std::vector<Eigen::VectorXd> &state_task,
        const std::vector<Eigen::VectorXd> &ctrl_task)
    {
        std::vector<boost::shared_ptr<ActionModelAbstract>> models;
        running_dams_.clear();
        models.push_back(createArrivalModel(
            covariance_.enabled ? covariance_.p0 : Eigen::VectorXd()));

        std::vector<boost::shared_ptr<ActionModelAbstract>> steps;
        for (std::size_t i = 0; i < n_knots; ++i)
        {
            if (i >= state_task.size())
            {
                throw std::runtime_error("not enough state task rows");
            }
            if (i < ctrl_task.size())
            {
                steps.push_back(createDiscreteModel(
                    timestep, state_task[i], ctrl_task[i],
                    /*is_terminal=*/i + 1 == n_knots));
            }
            else
            {
                steps.push_back(
                    createDiscreteModel(timestep, state_task[i],
                                        Eigen::VectorXd(),
                                        /*is_terminal=*/i + 1 == n_knots));
            }
        }

        models.insert(models.end(), steps.begin(), steps.end() - 1);
        return boost::make_shared<crocoddyl::ShootingProblem>(x0, models,
                                                              steps.back());
    }

    boost::shared_ptr<ActionModelAbstract> createDiscreteModel(
        double timestep, const Eigen::VectorXd &state_task,
        const Eigen::VectorXd &ctrl_task, const bool is_terminal = false)
    {
        boost::shared_ptr<CostModelSum> cost_model =
            boost::make_shared<CostModelSum>(state_, actuation_->get_nu());

        if (state_task.size() > 0 && state_task.array().allFinite())
        {
            Eigen::VectorXd w(state_->get_ndx());
            double coefficient = 1e1;
            if (covariance_.enabled)
            {
                w = covariance_.r;
                // IntegratedActionModelEuler multiplies controlled running
                // costs by dt, while terminal calc(x) does not.  Compensate so
                // both are exactly 0.5*r'R^-1*r per sample.
                coefficient = is_terminal ? 1.0 : 1.0 / timestep;
            }
            else
            {
                const int nv = rmodel_.nv;
                w.segment(0, 2).fill(std::pow(weights_.meas_position, 2));
                w.segment(2, 1).fill(std::pow(weights_.meas_position_z, 2));
                w.segment(3, 3).fill(std::pow(weights_.meas_orientation, 2));
                w.segment(6, nv - 6).fill(std::pow(weights_.meas_joint, 2));
                w.segment(nv, 3).fill(std::pow(weights_.meas_linearVel, 2));
                w.segment(nv + 3, 3).fill(std::pow(weights_.meas_angularVel, 2));
                w.segment(nv + 6, nv - 6).fill(std::pow(weights_.meas_jointVel, 2));
            }

            auto activation = boost::make_shared<
                crocoddyl::ActivationModelWeightedQuadTpl<double>>(w);
            auto cost = boost::make_shared<crocoddyl::CostModelResidualTpl<double>>(
                state_, activation,
                boost::make_shared<crocoddyl::ResidualModelStateTpl<double>>(
                    state_, state_task, actuation_->get_nu()));
            cost_model->addCost("stateReg", cost, coefficient);
        }

        if (ctrl_task.size() > 0 && ctrl_task.array().allFinite())
        {
            Eigen::VectorXd w(actuation_->get_nu());
            double coefficient = 1e1;
            if (covariance_.enabled)
            {
                if (is_terminal)
                    throw std::runtime_error("terminal covariance action has control task");
                w = covariance_.q;
                coefficient = 1.0 / timestep;
            }
            else
            {
                w.segment(0, 3).fill(std::pow(weights_.dyn_position, 2));
                w.segment(3, 3).fill(std::pow(weights_.dyn_orientation, 2));
                w.segment(6, actuation_->get_nu() - 6)
                    .fill(std::pow(weights_.dyn_joint, 2));
            }

            auto activation = boost::make_shared<
                crocoddyl::ActivationModelWeightedQuadTpl<double>>(w);
            auto cost = boost::make_shared<crocoddyl::CostModelResidualTpl<double>>(
                state_, activation,
                boost::make_shared<crocoddyl::ResidualModelControlTpl<double>>(
                    state_, ctrl_task));
            cost_model->addCost("ctrlReg", cost, coefficient);
        }

        auto dmodel =
            boost::make_shared<DifferentialActionModelContactFwdDynamicsMotion>(
                state_, actuation_, cost_model, contact_frame_names_,
                weights_.kappa, weights_.mu, timestep, true,
                newton_max_iters_);
        running_dams_.push_back(dmodel);

        return boost::make_shared<crocoddyl::IntegratedActionModelEulerTpl<double>>(
            dmodel, timestep);
    }

    // Arrival with an explicit 70-dim quadratic control cost.  In legacy mode
    // the precision weights are zero: the official arrival provides no motion
    // P0, and any nonzero value here would be an undeclared covariance claim.
    boost::shared_ptr<ActionModelAbstract> createArrivalModel(
        const Eigen::VectorXd &p0_precision_diag = Eigen::VectorXd())
    {
        const std::size_t ndx = state_->get_ndx();
        Eigen::VectorXd w = Eigen::VectorXd::Zero(ndx);
        if (p0_precision_diag.size() > 0)
        {
            if (static_cast<std::size_t>(p0_precision_diag.size()) != ndx)
            {
                throw std::runtime_error("p0 precision diag must be ndx");
            }
            w = p0_precision_diag;
        }

        auto cost_model = boost::make_shared<CostModelSum>(
            state_, static_cast<std::size_t>(ndx));
        auto activation = boost::make_shared<
            crocoddyl::ActivationModelWeightedQuadTpl<double>>(w);
        auto cost = boost::make_shared<crocoddyl::CostModelResidualTpl<double>>(
            state_, activation,
            boost::make_shared<crocoddyl::ResidualModelControlTpl<double>>(
                state_, Eigen::VectorXd::Zero(ndx)));
        // Coefficient 1 (not the legacy 10): this cost is exactly
        // 0.5 * sum w_i u_i^2 under Crocoddyl's quadratic convention.
        cost_model->addCost("arrivalReg", cost, 1.);

        auto arrival =
            boost::make_shared<ActionModelMotionArrival>(state_, cost_model);
        arrival_ = arrival;
        return arrival;
    }

    const boost::shared_ptr<StateMultibody> &get_state() const { return state_; }
    const std::vector<
        boost::shared_ptr<DifferentialActionModelContactFwdDynamicsMotion>> &
    get_running_dams() const
    {
        return running_dams_;
    }
    const boost::shared_ptr<ActionModelMotionArrival> &get_arrival() const
    {
        return arrival_;
    }

private:
    pinocchio::Model rmodel_;
    std::vector<std::string> contact_frame_names_;
    boost::shared_ptr<StateMultibody> state_;
    boost::shared_ptr<crocoddyl::ActuationModelAbstractTpl<double>> actuation_;
    crocoddyl::ContactIDWeights weights_;
    DiagonalCovariancePrecision covariance_;
    int newton_max_iters_;
    std::vector<boost::shared_ptr<DifferentialActionModelContactFwdDynamicsMotion>>
        running_dams_;
    boost::shared_ptr<ActionModelMotionArrival> arrival_;
};

} // namespace g1cal
