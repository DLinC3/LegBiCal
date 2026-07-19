///////////////////////////////////////////////////////////////////////////////
// g1cal motion-only overlay.
//
// ActionModelMotionArrival is the zero-duration arrival map of the motion-only
// FIE: from the fixed shooting anchor x (the measured prior state) it produces
//   xnext = integrate(x, delta_x),   delta_x = u in R^70.
// Cost is whatever CostModelSum the builder attaches (legacy parity: zero
// weights on u; covariance mode: 0.5 * delta_x' P0^{-1} delta_x).  There is no
// inertia slot anywhere: nu equals the full 70 state tangent.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <stdexcept>
#include <string>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace g1cal
{

class ActionDataMotionArrival;

class ActionModelMotionArrival
    : public crocoddyl::ActionModelAbstractTpl<double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef double Scalar;
    typedef crocoddyl::ActionModelAbstractTpl<double> Base;
    typedef ActionDataMotionArrival Data;
    typedef crocoddyl::ActionDataAbstractTpl<double> ActionDataAbstract;
    typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
    typedef crocoddyl::CostModelSumTpl<double> CostModelSum;
    typedef Eigen::VectorXd VectorXs;
    typedef Eigen::MatrixXd MatrixXs;

    ActionModelMotionArrival(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<CostModelSum> costs)
        : Base(state, state->get_ndx(), costs->get_nr()),
          motion_state_(state),
          costs_(costs)
    {
        if (costs_->get_nu() != nu_)
        {
            throw std::invalid_argument(
                "arrival costs control dimension != " + std::to_string(nu_));
        }
    }

    virtual ~ActionModelMotionArrival() {}

    virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x);

    virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x);

    virtual boost::shared_ptr<ActionDataAbstract> createData();
    virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data);

    virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract> &,
                             Eigen::Ref<VectorXs>,
                             const Eigen::Ref<const VectorXs> &,
                             const std::size_t = 100, const Scalar = 1e-9)
    {
        // Zero-duration arrival: no quasi-static torque concept.
    }

    const boost::shared_ptr<CostModelSum> &get_costs() const { return costs_; }
    const boost::shared_ptr<StateMultibody> &get_motion_state() const
    {
        return motion_state_;
    }

    virtual void print(std::ostream &os) const
    {
        os << "ActionModelMotionArrival {nx=" << state_->get_nx()
           << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << "}";
    }

protected:
    using Base::nu_;
    using Base::state_;

private:
    boost::shared_ptr<StateMultibody> motion_state_;
    boost::shared_ptr<CostModelSum> costs_;
};

class ActionDataMotionArrival
    : public crocoddyl::ActionDataAbstractTpl<double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef crocoddyl::ActionDataAbstractTpl<double> Base;

    explicit ActionDataMotionArrival(ActionModelMotionArrival *const model)
        : Base(model), collector()
    {
        costs = model->get_costs()->createData(&collector);
        costs->shareMemory(this);
    }

    crocoddyl::DataCollectorAbstractTpl<double> collector;
    boost::shared_ptr<crocoddyl::CostDataSumTpl<double>> costs;

    using Base::cost;
    using Base::Fu;
    using Base::Fx;
    using Base::xnext;
};

inline void ActionModelMotionArrival::calc(
    const boost::shared_ptr<ActionDataAbstract> &data,
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

    state_->integrate(x, u, d->xnext);

    costs_->calc(d->costs, x, u);
    d->cost = d->costs->cost;
}

inline void ActionModelMotionArrival::calc(
    const boost::shared_ptr<ActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x)
{
    Data *d = static_cast<Data *>(data.get());
    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
}

inline void ActionModelMotionArrival::calcDiff(
    const boost::shared_ptr<ActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
{
    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
        throw std::invalid_argument("x has wrong dimension");
    }
    Data *d = static_cast<Data *>(data.get());

    state_->Jintegrate(x, u, d->Fx, d->Fx, crocoddyl::first, crocoddyl::setto);
    MatrixXs Fu_full = MatrixXs::Zero(state_->get_ndx(), state_->get_ndx());
    state_->Jintegrate(x, u, Fu_full, Fu_full, crocoddyl::second,
                       crocoddyl::setto);
    d->Fu = Fu_full;

    costs_->calcDiff(d->costs, x, u);
    d->Lx = d->costs->Lx;
    d->Lu = d->costs->Lu;
    d->Lxx = d->costs->Lxx;
    d->Luu = d->costs->Luu;
    d->Lxu = d->costs->Lxu;
}

inline void ActionModelMotionArrival::calcDiff(
    const boost::shared_ptr<ActionDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x)
{
    Data *d = static_cast<Data *>(data.get());
    costs_->calcDiff(d->costs, x);
    d->Lx = d->costs->Lx;
    d->Lu = d->costs->Lu;
    d->Lxx = d->costs->Lxx;
    d->Luu = d->costs->Luu;
    d->Lxu = d->costs->Lxu;
}

inline boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<double>>
ActionModelMotionArrival::createData()
{
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

inline bool ActionModelMotionArrival::checkData(
    const boost::shared_ptr<ActionDataAbstract> &data)
{
    return boost::dynamic_pointer_cast<Data>(data) != nullptr;
}

} // namespace g1cal
