///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026 Jiarong Kang
//
// Developed at the Legged AI Lab, University of Wisconsin-Madison.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_PREPROCESS_HPP_
#define CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_PREPROCESS_HPP_

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "crocoddyl/contact_id/anitescu/Logchol.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "crocoddyl/multibody/utils/csv_to_eigen.hpp"

namespace contact_id_experiment {

struct PreparedContactIDData {
  // Initial state passed to the shooting problem. It is assembled from the
  // measured q/v row and model-derived inertial parameters.
  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> state_task;
  std::vector<Eigen::VectorXd> ctrl_task;
  std::vector<Eigen::VectorXd> state_init;
  std::vector<Eigen::VectorXd> ctrl_init;
  std::vector<Eigen::VectorXd> parameter_logs;
};

inline Eigen::VectorXd csv_row(const Eigen::MatrixXd& mat, std::size_t row,
                               bool has_time_column, Eigen::Index cols) {
  // The experiment CSVs are numeric and may carry a leading timestamp column.
  // The optimizer only consumes the state/control payload.
  const Eigen::Index offset = has_time_column ? 1 : 0;
  if (row >= static_cast<std::size_t>(mat.rows()) ||
      mat.cols() < offset + cols) {
    throw std::runtime_error("CSV does not contain the requested row/columns.");
  }
  return mat.row(static_cast<Eigen::Index>(row)).segment(offset, cols).transpose();
}

inline Eigen::VectorXd make_control(const Eigen::MatrixXd& us,
                                    std::size_t row0,
                                    std::size_t down_sample,
                                    bool has_time_column, Eigen::Index nv,
                                    bool average_controls) {
  // Controls may be averaged over each down-sampled knot interval, or sampled
  // from the first row to match older benchmark scripts. The contact-ID
  // actuation is floating-base, so the first six generalized force entries are
  // left zero and CSV actuator torques are placed at segment(6, nv - 6).
  // Some local logs contain extra columns after the actuator block; those are
  // ignored to match the original benchmark preprocessing.
  const Eigen::Index offset = has_time_column ? 1 : 0;
  const Eigen::Index raw_cols = us.cols() - offset;
  if (raw_cols < nv - 6) {
    throw std::runtime_error(
        "Control CSV must contain at least nv-6 actuator columns.");
  }
  Eigen::VectorXd ctrl = Eigen::VectorXd::Zero(nv);
  const std::size_t rows_to_use = average_controls ? down_sample : 1;
  for (std::size_t r = row0; r < row0 + rows_to_use; ++r) {
    if (r >= static_cast<std::size_t>(us.rows())) {
      throw std::runtime_error("Control CSV is shorter than requested horizon.");
    }
    ctrl.segment(6, nv - 6).noalias() +=
        us.row(static_cast<Eigen::Index>(r))
            .segment(offset, nv - 6)
            .transpose();
  }
  ctrl /= static_cast<double>(rows_to_use);
  return ctrl;
}

inline void apply_joint_order(Eigen::MatrixXd& mat, bool has_time_column,
                              Eigen::Index first_joint_col,
                              Eigen::Index joint_cols,
                              const std::vector<int>& order) {
  // Some robot logs use a simulator/controller joint order that differs from
  // the Pinocchio model. The optional XML permutation maps each model-order
  // joint column i to the raw log column order[i].
  if (order.empty()) {
    return;
  }
  if (static_cast<Eigen::Index>(order.size()) != joint_cols) {
    throw std::runtime_error("joint_order size must match the actuated joints.");
  }

  const Eigen::Index offset = has_time_column ? 1 : 0;
  const Eigen::Index start = offset + first_joint_col;
  if (mat.cols() < start + joint_cols) {
    throw std::runtime_error("CSV does not contain enough joint columns.");
  }

  Eigen::MatrixXd raw = mat.block(0, start, mat.rows(), joint_cols);
  for (Eigen::Index i = 0; i < joint_cols; ++i) {
    const int src = order[static_cast<std::size_t>(i)];
    if (src < 0 || src >= joint_cols) {
      throw std::runtime_error("joint_order contains an out-of-range index.");
    }
    mat.col(start + i) = raw.col(src);
  }
}

inline void normalize_quaternion(Eigen::VectorXd& x, Eigen::Index nq) {
  if (nq >= 7) {
    x.segment<4>(3).normalize();
  }
}

inline double lowest_contact_height(const pinocchio::Model& model,
                                    pinocchio::Data& data,
                                    const std::vector<std::string>& frames,
                                    const Eigen::VectorXd& q) {
  // Computes the world-z height of the lowest configured contact frame for one
  // configuration. The G1 benchmark subtracts this from each reference row so
  // the lowest configured contact is projected onto the terrain.
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  double min_z = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const pinocchio::FrameIndex fid = model.getFrameId(frames[i]);
    if (fid >= static_cast<pinocchio::FrameIndex>(model.nframes)) {
      throw std::runtime_error("Contact frame not found in model: " +
                               frames[i]);
    }
    min_z = std::min(min_z, data.oMf[fid].translation().z());
  }
  return min_z;
}

inline void apply_offsets(Eigen::VectorXd& params,
                          const std::vector<Eigen::VectorXd>& offsets,
                          std::size_t joint_index) {
  if (joint_index >= offsets.size() || offsets[joint_index].size() == 0) {
    return;
  }
  if (offsets[joint_index].size() != params.size()) {
    throw std::runtime_error("Identification offset must have 10 entries.");
  }
  params += offsets[joint_index];
}

inline pinocchio::SE3 se3_from_xyz_quat_xyzw(const Eigen::Vector3d& p,
                                             const Eigen::Vector4d& q_xyzw) {
  Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
  q.normalize();
  return pinocchio::SE3(q.toRotationMatrix(), p);
}

inline Eigen::Vector4d quat_xyzw_from_se3(const pinocchio::SE3& transform) {
  Eigen::Quaterniond q(transform.rotation());
  q.normalize();
  return Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());
}

inline void convert_torso_measurements_to_base(
    const pinocchio::Model& model, const std::string& torso_frame_name,
    bool has_time_column, Eigen::MatrixXd& q_csv) {
  // Mocap may log a torso/body pose where Pinocchio expects the free-flyer base.
  // Given the logged joint angles, recover world_M_base from world_M_torso and
  // the model transform base_M_torso.
  if (torso_frame_name.empty()) {
    return;
  }
  const Eigen::Index offset = has_time_column ? 1 : 0;
  if (q_csv.cols() < offset + model.nq) {
    throw std::runtime_error(
        "q CSV does not contain enough columns for torso-base conversion.");
  }
  const pinocchio::FrameIndex torso_fid =
      model.getFrameId(torso_frame_name, pinocchio::BODY);
  if (torso_fid >= static_cast<pinocchio::FrameIndex>(model.nframes)) {
    throw std::runtime_error("torso_base_frame not found in model: " +
                             torso_frame_name);
  }

  pinocchio::Data data(model);
  Eigen::VectorXd q_fk = Eigen::VectorXd::Zero(model.nq);
  q_fk.segment<3>(0).setZero();
  q_fk.segment<4>(3) << 0., 0., 0., 1.;

  for (Eigen::Index k = 0; k < q_csv.rows(); ++k) {
    const Eigen::Vector3d world_p_torso =
        q_csv.row(k).segment(offset, 3).transpose();
    const Eigen::Vector4d world_q_torso_xyzw =
        q_csv.row(k).segment(offset + 3, 4).transpose();
    const pinocchio::SE3 world_M_torso =
        se3_from_xyz_quat_xyzw(world_p_torso, world_q_torso_xyzw);

    q_fk.tail(model.nq - 7) =
        q_csv.row(k).segment(offset + 7, model.nq - 7).transpose();
    pinocchio::forwardKinematics(model, data, q_fk);
    pinocchio::updateFramePlacements(model, data);

    const pinocchio::SE3 pelvis_M_torso = data.oMf[torso_fid];
    const pinocchio::SE3 world_M_pelvis =
        world_M_torso * pelvis_M_torso.inverse();

    q_csv.row(k).segment(offset, 3) =
        world_M_pelvis.translation().transpose();
    q_csv.row(k).segment(offset + 3, 4) =
        quat_xyzw_from_se3(world_M_pelvis).transpose();
  }
}

inline std::vector<Eigen::VectorXd> compute_parameter_logs(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const std::vector<Eigen::VectorXd>& offsets) {
  // Identification starts from the model inertial parameters represented in
  // log-Cholesky coordinates, with optional XML offsets for local experiments.
  std::vector<Eigen::VectorXd> pi_logs;
  for (std::size_t j = 0; j < joints.size(); ++j) {
    Eigen::VectorXd pi_log = computeLogCholeskyFromLink(
        model, static_cast<int>(joints[j]));
    apply_offsets(pi_log, offsets, j);
    pi_logs.push_back(pi_log);
  }
  return pi_logs;
}

inline void fill_state(const pinocchio::Model& model,
                       const std::vector<Eigen::VectorXd>& pi_logs,
                       const crocoddyl::ContactIDDataConfig& data_cfg,
                       const Eigen::MatrixXd& q_csv,
                       const Eigen::MatrixXd& v_csv, std::size_t row,
                       Eigen::VectorXd& x) {
  // Fill the Crocoddyl contact-ID state layout:
  //   [q, identified inertial log params, v, parameter-rate slots (zeros)].
  // The trailing parameter-rate slots stay at their existing/default values.
  x.segment(0, model.nq) =
      csv_row(q_csv, row, data_cfg.q_has_time_column, model.nq);
  if (data_cfg.normalize_base_quaternion) {
    normalize_quaternion(x, model.nq);
  }
  for (std::size_t j = 0; j < pi_logs.size(); ++j) {
    x.segment(model.nq + static_cast<Eigen::Index>(10 * j), 10) = pi_logs[j];
  }
  x.segment(model.nq + static_cast<Eigen::Index>(10 * pi_logs.size()),
            model.nv) =
      csv_row(v_csv, row, data_cfg.v_has_time_column, model.nv);
}

inline PreparedContactIDData prepare_contact_id_data(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const std::vector<std::string>& contact_frames,
    const crocoddyl::ContactIDXMLConfig& cfg) {
  PreparedContactIDData prepared;
  prepared.parameter_logs =
      compute_parameter_logs(model, joints, cfg.parameter_offsets);

  const Eigen::MatrixXd q_csv = csvutil::readCSVtoEigen(cfg.data.q_csv);
  Eigen::MatrixXd q_processed = q_csv;
  Eigen::MatrixXd v_processed = csvutil::readCSVtoEigen(cfg.data.v_csv);
  Eigen::MatrixXd u_processed = csvutil::readCSVtoEigen(cfg.data.u_csv);

  apply_joint_order(q_processed, cfg.data.q_has_time_column, 7, model.nq - 7,
                    cfg.data.joint_order);
  apply_joint_order(v_processed, cfg.data.v_has_time_column, 6, model.nv - 6,
                    cfg.data.joint_order);
  apply_joint_order(u_processed, cfg.data.u_has_time_column, 0, model.nv - 6,
                    cfg.data.joint_order);

  convert_torso_measurements_to_base(model, cfg.data.torso_base_frame,
                                     cfg.data.q_has_time_column, q_processed);

  const std::size_t last_state_row = cfg.solver.start_idx + cfg.solver.horizon;
  if (last_state_row > static_cast<std::size_t>(q_processed.rows()) ||
      last_state_row > static_cast<std::size_t>(v_processed.rows())) {
    throw std::runtime_error("State CSVs are shorter than requested horizon.");
  }

  pinocchio::Data pin_data(model);

  const Eigen::Index nx =
      model.nq + model.nv +
      static_cast<Eigen::Index>(2 * 10 * prepared.parameter_logs.size());

  // The contact-ID state layout is determined by the Pinocchio model and the
  // number of identified links. Unfilled parameter-rate slots remain zero.
  prepared.x0 = Eigen::VectorXd::Zero(nx);
  fill_state(model, prepared.parameter_logs, cfg.data, q_processed, v_processed,
             cfg.solver.start_idx, prepared.x0);

  double initial_ground_height = 0.;
  if (cfg.data.shift_base_to_ground) {
    initial_ground_height =
        lowest_contact_height(model, pin_data, contact_frames,
                              prepared.x0.head(model.nq));
    prepared.x0[2] -= initial_ground_height;
  }

  prepared.state_init.push_back(prepared.x0);
  prepared.ctrl_init.push_back(Eigen::VectorXd::Zero(10 * joints.size()));

  for (std::size_t i = 0; i < cfg.solver.horizon;
       i += cfg.solver.down_sample) {
    const std::size_t row = cfg.solver.start_idx + i;
    Eigen::VectorXd x_i = Eigen::VectorXd::Zero(nx);
    fill_state(model, prepared.parameter_logs, cfg.data, q_processed, v_processed,
               row, x_i);
    if (cfg.data.shift_base_to_ground) {
      const double ground_height =
          cfg.data.reuse_initial_ground_height
              ? initial_ground_height
              : lowest_contact_height(model, pin_data, contact_frames,
                                      x_i.head(model.nq));
      x_i[2] -= ground_height;
    }
    prepared.state_task.push_back(x_i);
    prepared.state_init.push_back(x_i);
  }

  for (std::size_t i = 0; i + cfg.solver.down_sample < cfg.solver.horizon;
       i += cfg.solver.down_sample) {
    const std::size_t row = cfg.solver.start_idx + i;
    Eigen::VectorXd u_i =
        make_control(u_processed, row, cfg.solver.down_sample,
                     cfg.data.u_has_time_column, model.nv,
                     cfg.data.average_controls);
    prepared.ctrl_task.push_back(u_i);
    prepared.ctrl_init.push_back(u_i);
  }

  return prepared;
}

}  // namespace contact_id_experiment

#endif  // CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_PREPROCESS_HPP_
