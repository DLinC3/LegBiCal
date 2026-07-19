///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026 Jiarong Kang
//
// Developed at the Legged AI Lab, University of Wisconsin-Madison.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_OUTPUTS_HPP_
#define CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_OUTPUTS_HPP_

#include <Eigen/Dense>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/multibody/utils/csv_to_eigen.hpp"

#include <pinocchio/spatial/inertia.hpp>

namespace contact_id_experiment {

inline void ensure_dir(const std::string& path) {
  if (path.empty() || path == ".") {
    return;
  }
  std::string partial;
  for (std::size_t i = 0; i < path.size(); ++i) {
    partial.push_back(path[i]);
    if (path[i] != '/' && i + 1 != path.size()) {
      continue;
    }
    if (partial.empty() || partial == "/") {
      continue;
    }
    if (::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
      throw std::runtime_error("Failed to create directory '" + partial +
                               "': " + std::strerror(errno));
    }
  }
}

inline void truncate_file(const std::string& path) {
  if (path.empty()) {
    return;
  }
  std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open output file: " + path);
  }
}

inline std::string output_path(const crocoddyl::ContactIDOutputConfig& outputs,
                               const std::string& filename) {
  return crocoddyl::contact_id_xml::join_path(outputs.directory, filename);
}

inline void clear_outputs(const crocoddyl::ContactIDOutputConfig& outputs) {
  // Truncate configured logs before a run so stale rows from a previous horizon
  // cannot be mistaken for the current solution.
  ensure_dir(outputs.directory);
  truncate_file(output_path(outputs, outputs.xs_results));
  truncate_file(output_path(outputs, outputs.us_results));
  truncate_file(output_path(outputs, outputs.u0_results));
  if (!outputs.inertia_report.empty()) {
    truncate_file(output_path(outputs, outputs.inertia_report));
  }
  if (outputs.rollout) {
    truncate_file(output_path(outputs, outputs.xs_rollout));
  }
  if (!outputs.force_log.empty()) {
    truncate_file(output_path(outputs, outputs.force_log));
  }
  if (outputs.log_initial_guess) {
    truncate_file(output_path(outputs, outputs.xs_log));
    truncate_file(output_path(outputs, outputs.us_log));
  }
}

inline std::string force_log_path(
    const crocoddyl::ContactIDOutputConfig& outputs) {
  if (outputs.force_log.empty()) {
    return std::string();
  }
  return output_path(outputs, outputs.force_log);
}

inline void log_initial_guess(
    const crocoddyl::ContactIDOutputConfig& outputs,
    const std::vector<Eigen::VectorXd>& state_task,
    const std::vector<Eigen::VectorXd>& ctrl_task) {
  // These logs mirror the preprocessed reference trajectory used by the
  // optimization. They are useful for checking CSV formatting and frame
  // conversion.
  if (!outputs.log_initial_guess) {
    return;
  }
  for (std::size_t i = 0; i < state_task.size(); ++i) {
    csvutil::logVectorToCSV(state_task[i], output_path(outputs, outputs.xs_log));
  }
  for (std::size_t i = 0; i < ctrl_task.size(); ++i) {
    csvutil::logVectorToCSV(ctrl_task[i], output_path(outputs, outputs.us_log));
  }
}

inline void save_vector_sequence_csv(
    const crocoddyl::ContactIDOutputConfig& outputs,
    const std::string& filename,
    const std::vector<Eigen::VectorXd>& values) {
  if (filename.empty() || values.empty()) {
    return;
  }
  ensure_dir(outputs.directory);
  const std::string path = output_path(outputs, filename);
  truncate_file(path);
  for (std::size_t i = 0; i < values.size(); ++i) {
    csvutil::logVectorToCSV(values[i], path);
  }
}

inline void save_solution_outputs(
    const crocoddyl::ContactIDOutputConfig& outputs,
    const boost::shared_ptr<crocoddyl::ShootingProblem>& shooting_problem,
    const std::vector<Eigen::VectorXd>& xs_sol,
    const std::vector<Eigen::VectorXd>& us_sol) {
  // Keep all file emission here so action models and problem builders stay free
  // of experiment-path side effects.
  if (!xs_sol.empty()) {
    Eigen::MatrixXd xs_results(xs_sol.size(), xs_sol[0].size());
    for (std::size_t i = 0; i < xs_sol.size(); ++i) {
      xs_results.row(static_cast<Eigen::Index>(i)) = xs_sol[i].transpose();
    }
    csvutil::saveEigenToCSV(output_path(outputs, outputs.xs_results),
                            xs_results);
  }

  if (us_sol.size() > 1) {
    Eigen::MatrixXd us_results(us_sol.size() - 1, us_sol[1].size());
    us_results.setZero();
    for (std::size_t i = 1; i < us_sol.size(); ++i) {
      us_results.row(static_cast<Eigen::Index>(i - 1)) =
          us_sol[i].transpose();
    }
    csvutil::saveEigenToCSV(output_path(outputs, outputs.us_results),
                            us_results);
  }

  if (!us_sol.empty()) {
    Eigen::MatrixXd u0_mat(1, us_sol[0].size());
    u0_mat.row(0) = us_sol[0].transpose();
    csvutil::saveEigenToCSV(output_path(outputs, outputs.u0_results), u0_mat);
  }

  if (outputs.rollout) {
    std::vector<Eigen::VectorXd> xs_rollout(shooting_problem->get_T() + 1);
    shooting_problem->rollout(us_sol, xs_rollout);
    Eigen::MatrixXd xs_roll(xs_rollout.size(), xs_rollout[0].size());
    for (std::size_t i = 0; i < xs_rollout.size(); ++i) {
      xs_roll.row(static_cast<Eigen::Index>(i)) = xs_rollout[i].transpose();
    }
    csvutil::saveEigenToCSV(output_path(outputs, outputs.xs_rollout), xs_roll);
  }
}

template <typename Derived>
inline void write_vector(std::ostream& os, const std::string& label,
                         const Eigen::MatrixBase<Derived>& v) {
  static const Eigen::IOFormat fmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[",
      "]");
  os << label << " = " << v.transpose().format(fmt) << "\n";
}

template <typename Derived>
inline void write_matrix(std::ostream& os, const std::string& label,
                         const Eigen::MatrixBase<Derived>& m) {
  static const Eigen::IOFormat fmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]", "",
      "");
  os << label << ":\n" << m.format(fmt) << "\n";
}

inline pinocchio::Inertia inertia_from_log_cholesky(
    const Eigen::VectorXd& log_params) {
  if (log_params.size() != 10) {
    throw std::runtime_error("Expected 10 log-Cholesky inertia parameters.");
  }
  Eigen::Matrix<double, 10, 1> fixed_log_params = log_params;
  pinocchio::LogCholeskyParametersTpl<double> log_chol(fixed_log_params);
  return pinocchio::Inertia::FromDynamicParameters(
      log_chol.toDynamicParameters());
}

inline void write_inertia_block(std::ostream& os, const std::string& label,
                                const pinocchio::Inertia& inertia) {
  const Eigen::Matrix<double, 10, 1> dynamic_params =
      inertia.toDynamicParameters();
  const double mass = inertia.mass();
  const Eigen::Vector3d first_moment = dynamic_params.segment<3>(1);
  Eigen::Vector3d com;
  if (std::abs(mass) > std::numeric_limits<double>::epsilon()) {
    com = first_moment / mass;
  } else {
    com = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
  }
  const pinocchio::PseudoInertia pseudo_inertia =
      inertia.toPseudoInertia();

  os << "\n--- " << label << " ---\n";
  os << "mass = " << mass << "\n";
  write_vector(os, "first_moment_h_m_com", first_moment);
  write_vector(os, "com_in_inertia_frame", com);
  write_vector(os, "dynamic_parameters", dynamic_params);
  write_matrix(os, "inertia_about_frame_origin", inertia.inertia().matrix());
  write_matrix(os, "pseudo_inertia", pseudo_inertia.toMatrix());
}

inline void save_inertia_identification_report(
    const crocoddyl::ContactIDOutputConfig& outputs,
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& identified_joints,
    const std::vector<Eigen::VectorXd>& initial_log_params,
    const std::vector<Eigen::VectorXd>& xs_sol,
    const std::vector<Eigen::VectorXd>& us_sol) {
  if (outputs.inertia_report.empty()) {
    return;
  }
  if (identified_joints.size() != initial_log_params.size()) {
    throw std::runtime_error(
        "Cannot write inertia report: identified joint and parameter counts "
        "do not match.");
  }

  std::ofstream os(output_path(outputs, outputs.inertia_report).c_str(),
                   std::ios::out | std::ios::trunc);
  if (!os.is_open()) {
    throw std::runtime_error("Cannot open inertia report output file: " +
                             output_path(outputs, outputs.inertia_report));
  }
  os << std::setprecision(12);
  os << "PRIME inertia-parameter identification report\n";
  os << "================================================\n\n";
  os << "model = " << model.name << "\n";
  os << "nq = " << model.nq << ", nv = " << model.nv << "\n";
  os << "identified_links = " << identified_joints.size() << "\n";
  os << "log_cholesky_order = "
        "[alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]\n";
  os << "dynamic_parameter_order = "
        "[mass, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]\n\n";

  const Eigen::VectorXd* estimated_state = NULL;
  if (xs_sol.size() > 1) {
    estimated_state = &xs_sol[1];
  } else if (!xs_sol.empty()) {
    estimated_state = &xs_sol[0];
  }

  for (std::size_t i = 0; i < identified_joints.size(); ++i) {
    const pinocchio::JointIndex jid = identified_joints[i];
    const std::string joint_name =
        jid < model.names.size() ? model.names[jid] : std::string("<unknown>");

    os << "\n================================================\n";
    os << "identified_entry[" << i << "]\n";
    os << "joint_index = " << jid << "\n";
    os << "joint_name = " << joint_name << "\n";

    const Eigen::VectorXd initial_log = initial_log_params[i];
    Eigen::VectorXd estimated_log = initial_log;
    if (estimated_state != NULL &&
        estimated_state->size() >=
            model.nq + static_cast<Eigen::Index>(10 * (i + 1))) {
      estimated_log = estimated_state->segment(
          model.nq + static_cast<Eigen::Index>(10 * i), 10);
    } else {
      os << "[warn] optimized state does not contain this parameter block; "
            "using the initial parameters as the estimate.\n";
    }

    write_vector(os, "initial_log_cholesky", initial_log);
    write_vector(os, "estimated_log_cholesky", estimated_log);
    write_vector(os, "estimated_minus_initial_log_cholesky",
                 estimated_log - initial_log);

    if (!us_sol.empty() &&
        us_sol[0].size() >= static_cast<Eigen::Index>(10 * (i + 1))) {
      write_vector(os, "u0_parameter_update",
                   us_sol[0].segment(static_cast<Eigen::Index>(10 * i), 10));
    }

    const pinocchio::Inertia initial_inertia =
        inertia_from_log_cholesky(initial_log);
    const pinocchio::Inertia estimated_inertia =
        inertia_from_log_cholesky(estimated_log);

    write_inertia_block(os, "initial inertia", initial_inertia);
    write_inertia_block(os, "estimated inertia", estimated_inertia);
  }
}

}  // namespace contact_id_experiment

#endif  // CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_OUTPUTS_HPP_
