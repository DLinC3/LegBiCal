///////////////////////////////////////////////////////////////////////////////
// g1cal motion-only overlay.
//
// DifferentialActionModelContactFwdDynamicsMotion mirrors the pinned
// DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl value path with
// the inertial-parameter machinery structurally removed:
//   - state is the ordinary crocoddyl::StateMultibody (nx=71, ndx=70 for G1);
//   - control is the 35 generalized-force vector;
//   - no state slice is interpreted as inertia; the Pinocchio model is const
//     and never mutated by calc()/calcDiff();
//   - contact value equations, offsets, regularizers, armature/damping, and
//     Newton behavior are those of the frozen simulator copy.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

#include "g1cal/motion_simulator.hpp"

namespace g1cal
{

class DifferentialActionDataContactFwdDynamicsMotion;

class DifferentialActionModelContactFwdDynamicsMotion
    : public crocoddyl::DifferentialActionModelAbstractTpl<double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef double Scalar;
    typedef crocoddyl::DifferentialActionModelAbstractTpl<double> Base;
    typedef DifferentialActionDataContactFwdDynamicsMotion Data;
    typedef crocoddyl::DifferentialActionDataAbstractTpl<double>
        DifferentialActionDataAbstract;
    typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
    typedef crocoddyl::CostModelSumTpl<double> CostModelSum;
    typedef crocoddyl::ActuationModelAbstractTpl<double> ActuationModelAbstract;
    typedef Eigen::VectorXd VectorXs;
    typedef Eigen::MatrixXd MatrixXs;

    DifferentialActionModelContactFwdDynamicsMotion(
        boost::shared_ptr<StateMultibody> state,
        boost::shared_ptr<ActuationModelAbstract> actuation,
        boost::shared_ptr<CostModelSum> costs,
        std::vector<std::string> contact_frames,
        const Scalar kappa, const Scalar mu, const Scalar dt,
        const bool exact_q_sensitivity = true,
        const int newton_max_iters = 300)
        : Base(state, actuation->get_nu(), costs->get_nr(), 0, 0),
          motion_state_(state),
          actuation_(actuation),
          costs_(costs),
          contact_frames_(contact_frames),
          kappa_(kappa),
          mu_(mu),
          dt_(dt),
          exact_q_sensitivity_(exact_q_sensitivity),
          newton_max_iters_(newton_max_iters)
    {
        if (costs_->get_nu() != nu_)
        {
            throw std::invalid_argument(
                "costs control dimension != " + std::to_string(nu_));
        }
        const auto &pin = *state->get_pinocchio();
        VectorXs u_lb = Scalar(-1.) * pin.effortLimit.tail(nu_);
        VectorXs u_ub = Scalar(+1.) * pin.effortLimit.tail(nu_);
        Base::set_u_lb(u_lb);
        Base::set_u_ub(u_ub);
    }

    virtual ~DifferentialActionModelContactFwdDynamicsMotion() {}

    virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x);

    virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x);

    virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

    virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract> &data);

    const boost::shared_ptr<ActuationModelAbstract> &get_actuation() const
    {
        return actuation_;
    }
    const boost::shared_ptr<CostModelSum> &get_costs() const { return costs_; }
    const boost::shared_ptr<StateMultibody> &get_motion_state() const
    {
        return motion_state_;
    }
    // Const Pinocchio model access only: the motion-only action never mutates
    // model inertias.
    const pinocchio::ModelTpl<Scalar> &get_pinocchio() const
    {
        return *motion_state_->get_pinocchio();
    }
    const std::vector<std::string> &get_contact_frames() const
    {
        return contact_frames_;
    }
    Scalar get_kappa() const { return kappa_; }
    Scalar get_mu() const { return mu_; }
    Scalar get_dt() const { return dt_; }
    bool get_exact_q_sensitivity() const { return exact_q_sensitivity_; }
    int get_newton_max_iters() const { return newton_max_iters_; }
    void reset_contact_warm_start()
    {
        sim_out_.reset();
    }

    // Diagnostics of the most recent contact solve (calc with control).
    const ContactStepDiag &get_last_contact_diag() const { return last_diag_; }
    // Latent contact force of the most recent calc, source order [t1,t2,n].
    const VectorXs &get_last_force() const { return last_force_; }

    virtual void print(std::ostream &os) const
    {
        os << "DifferentialActionModelContactFwdDynamicsMotion {nx="
           << state_->get_nx() << ", ndx=" << state_->get_ndx()
           << ", nu=" << nu_ << "}";
    }

protected:
    using Base::nu_;
    using Base::state_;

private:
    boost::shared_ptr<StateMultibody> motion_state_;
    boost::shared_ptr<ActuationModelAbstract> actuation_;
    boost::shared_ptr<CostModelSum> costs_;
    std::vector<std::string> contact_frames_;
    Scalar kappa_;
    Scalar mu_;
    Scalar dt_;
    bool exact_q_sensitivity_;
    int newton_max_iters_;

    std::unique_ptr<MotionStepResult> sim_out_;
    ContactStepDiag last_diag_;
    VectorXs last_force_;
};

class DifferentialActionDataContactFwdDynamicsMotion
    : public crocoddyl::DifferentialActionDataAbstractTpl<double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef double Scalar;
    typedef crocoddyl::DifferentialActionDataAbstractTpl<double> Base;
    typedef crocoddyl::JointDataAbstractTpl<double> JointDataAbstract;
    typedef crocoddyl::DataCollectorJointActMultibodyTpl<double>
        DataCollectorJointActMultibody;

    explicit DifferentialActionDataContactFwdDynamicsMotion(
        DifferentialActionModelContactFwdDynamicsMotion *const model)
        : Base(model),
          pinocchio(pinocchio::DataTpl<double>(model->get_pinocchio())),
          multibody(&pinocchio, model->get_actuation()->createData(),
                    boost::make_shared<JointDataAbstract>(
                        model->get_state(), model->get_actuation(),
                        model->get_nu())),
          costs(model->get_costs()->createData(&multibody))
    {
        multibody.joint->dtau_du.diagonal().setOnes();
        costs->shareMemory(this);
    }

    pinocchio::DataTpl<double> pinocchio;
    DataCollectorJointActMultibody multibody;
    boost::shared_ptr<crocoddyl::CostDataSumTpl<double>> costs;

    using Base::cost;
    using Base::Fu;
    using Base::Fx;
    using Base::r;
    using Base::xout;
};

inline void DifferentialActionModelContactFwdDynamicsMotion::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
{
    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
        throw std::invalid_argument("x has wrong dimension");
    }
    if (static_cast<std::size_t>(u.size()) != nu_)
    {
        throw std::invalid_argument("u has wrong dimension");
    }

    Data *d = static_cast<Data *>(data.get());
    const std::size_t nq = motion_state_->get_nq_pin();
    const std::size_t nv = motion_state_->get_nv_pin();

    const Eigen::VectorXd q = x.head(nq);
    const Eigen::VectorXd v = x.tail(nv);

    actuation_->calc(d->multibody.actuation, x, u);

    // Mirrors the frozen ID action: computeAllTerms on the action's own data,
    // then a freshly constructed simulator performs the contact step.  No
    // inertia is read from the state and the model reference is const.
    pinocchio::computeAllTerms(get_pinocchio(), d->pinocchio, q, v);

    MotionAnitescuSimulator::Options opts;
    opts.mu = mu_;
    opts.plane_height = 0.0;
    opts.kappa = kappa_;
    opts.exact_q_sensitivity = exact_q_sensitivity_;
    opts.newton_max_iters = newton_max_iters_;
    opts.robust_newton_refinement = exact_q_sensitivity_;

    Eigen::VectorXd contact_warm_start;
    const bool use_contact_warm_start =
        exact_q_sensitivity_ && sim_out_ &&
        sim_out_->v_next.size() == static_cast<Eigen::Index>(nv);
    if (use_contact_warm_start)
        contact_warm_start = sim_out_->v_next;

    // The simulator owns a full Pinocchio Data object and sizeable contact
    // workspaces.  It is scratch for this evaluation, not per-knot persistent
    // model state.  Retaining one simulator in every action scales to tens of
    // gigabytes on long FIE horizons.  calcDiff needs only MotionStepResult,
    // which remains stored below and also carries the next-velocity warm start.
    MotionAnitescuSimulator simulator(get_pinocchio(), contact_frames_, opts);

    const Eigen::VectorXd tau = d->multibody.actuation->tau;

    sim_out_ = std::make_unique<MotionStepResult>(simulator.step(
        q, v, tau, dt_,
        use_contact_warm_start ? &contact_warm_start : nullptr));
    last_diag_ = sim_out_->diag;
    last_force_ = sim_out_->force;

    d->xout = (sim_out_->v_next - v) / dt_;

    d->multibody.joint->a = d->xout;
    d->multibody.joint->tau = u;
    costs_->calc(d->costs, x, u);
    d->cost = d->costs->cost;
}

inline void DifferentialActionModelContactFwdDynamicsMotion::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x)
{
    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
        throw std::invalid_argument("x has wrong dimension");
    }
    Data *d = static_cast<Data *>(data.get());
    const std::size_t nq = motion_state_->get_nq_pin();
    const std::size_t nv = motion_state_->get_nv_pin();
    const Eigen::VectorXd q = x.head(nq);
    const Eigen::VectorXd v = x.tail(nv);

    pinocchio::computeAllTerms(get_pinocchio(), d->pinocchio, q, v);

    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
}

inline void DifferentialActionModelContactFwdDynamicsMotion::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
{
    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
        throw std::invalid_argument("x has wrong dimension");
    }
    if (static_cast<std::size_t>(u.size()) != nu_)
    {
        throw std::invalid_argument("u has wrong dimension");
    }
    if (!sim_out_)
    {
        throw std::runtime_error("calcDiff called before calc");
    }

    const std::size_t nv = motion_state_->get_nv_pin();
    Data *d = static_cast<Data *>(data.get());

    actuation_->calcDiff(d->multibody.actuation, x, u);

    // Continuous-time layout on the 70-tangent: columns [dq(35), dv(35)].
    d->Fx.block(0, 0, nv, nv).noalias() = sim_out_->dv_dq / dt_;
    d->Fx.block(0, nv, nv, nv).noalias() =
        (sim_out_->dv_dv - MatrixXs::Identity(nv, nv)) / dt_;

    d->Fu.topRows(nv).noalias() =
        sim_out_->dv_dtau * d->multibody.actuation->dtau_du / dt_;

    d->multibody.joint->da_dx = d->Fx;
    d->multibody.joint->da_du = d->Fu;

    costs_->calcDiff(d->costs, x, u);
}

inline void DifferentialActionModelContactFwdDynamicsMotion::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x)
{
    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
        throw std::invalid_argument("x has wrong dimension");
    }
    Data *d = static_cast<Data *>(data.get());
    costs_->calcDiff(d->costs, x);
}

inline boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double>>
DifferentialActionModelContactFwdDynamicsMotion::createData()
{
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

inline bool DifferentialActionModelContactFwdDynamicsMotion::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract> &data)
{
    return boost::dynamic_pointer_cast<Data>(data) != nullptr;
}

} // namespace g1cal
