///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>

#include "crocoddyl/multibody/residuals/frame-placement.hpp"

namespace crocoddyl
{

  template <typename Scalar>
  ResidualCircularObstacleTpl<Scalar>::ResidualCircularObstacleTpl(
      boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
      const VectorXs &center, const Scalar distance, const std::size_t nu)
      : Base(state, 1, nu, true, false, false),
        id_(id),
        center_(center),
        distance_(distance),
        pin_model_(state->get_pinocchio())
  {
    if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
        id)
    {
      throw_pretty(
          "Invalid argument: "
          << "the frame index is wrong (it does not exist in the robot)");
    }
  }

  template <typename Scalar>
  ResidualCircularObstacleTpl<Scalar>::ResidualCircularObstacleTpl(
      boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
      const VectorXs &center, const Scalar distance)
      : Base(state, 1, true, false, false),
        id_(id),
        center_(center),
        distance_(distance),
        pin_model_(state->get_pinocchio())
  {
    if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
        id)
    {
      throw_pretty(
          "Invalid argument: "
          << "the frame index is wrong (it does not exist in the robot)");
    }
  }

  template <typename Scalar>
  ResidualCircularObstacleTpl<Scalar>::~ResidualCircularObstacleTpl() {}

  template <typename Scalar>
  void ResidualCircularObstacleTpl<Scalar>::calc(
      const boost::shared_ptr<ResidualDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &)
  {
    Data *d = static_cast<Data *>(data.get());
    // Compute the frame placement w.r.t. the reference frame
    pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
    VectorXs p = d->pinocchio->oMf[id_].translation();
    Scalar r = (p - center_).transpose() * (p - center_);
    Scalar lambda = Scalar(0.4);
    Scalar kappa = Scalar(0.4);
    data->r(0) = lambda * std::exp(-kappa * (r - distance_ * distance_));
  }

  template <typename Scalar>
  void ResidualCircularObstacleTpl<Scalar>::calcDiff(
      const boost::shared_ptr<ResidualDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &)
  {
    Data *d = static_cast<Data *>(data.get());

    // Compute the derivatives of the frame placement
    const std::size_t nv = state_->get_nv();
    pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                                pinocchio::LOCAL_WORLD_ALIGNED, d->fJf);

    VectorXs p = d->pinocchio->oMf[id_].translation();
    Scalar lambda = Scalar(0.4);
    Scalar kappa = Scalar(0.4);
    VectorXs Rp = -2 * kappa * data->r(0) * (p - center_);
    data->Rx.leftCols(nv).noalias() = Rp.transpose() * d->fJf.topRows(3);
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>>
  ResidualCircularObstacleTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                        data);
  }

  template <typename Scalar>
  void ResidualCircularObstacleTpl<Scalar>::print(std::ostream &os) const
  {
  }

  template <typename Scalar>
  pinocchio::FrameIndex ResidualCircularObstacleTpl<Scalar>::get_id() const
  {
    return id_;
  }

  template <typename Scalar>
  const pinocchio::SE3Tpl<Scalar> &
  ResidualCircularObstacleTpl<Scalar>::get_reference() const
  {
    return center_;
  }

  template <typename Scalar>
  void ResidualCircularObstacleTpl<Scalar>::set_id(
      const pinocchio::FrameIndex id)
  {
    id_ = id;
  }

  template <typename Scalar>
  void ResidualCircularObstacleTpl<Scalar>::set_reference(
      const VectorXs &center, const Scalar distance)
  {
    center_ = center;
    distance_ = distance;
  }

} // namespace crocoddyl
