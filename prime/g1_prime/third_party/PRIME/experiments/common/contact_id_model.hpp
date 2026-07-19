///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026 Jiarong Kang
//
// Developed at the Legged AI Lab, University of Wisconsin-Madison.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_MODEL_HPP_
#define CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_MODEL_HPP_

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "crocoddyl/contact_id/config/contact-id-config.hpp"

namespace contact_id_experiment {

inline void validate_required(const crocoddyl::ContactIDXMLConfig& cfg) {
  // Keep validation near the executable layer so XML errors are reported before
  // Pinocchio/Crocoddyl starts allocating models and solver objects.
  if (cfg.robot.urdf.empty()) {
    throw std::runtime_error("XML config requires <robot urdf=\"...\">.");
  }
  if (cfg.contact_frames.empty()) {
    throw std::runtime_error("XML config requires at least one contact frame.");
  }
  if (cfg.data.q_csv.empty() || cfg.data.v_csv.empty() ||
      cfg.data.u_csv.empty()) {
    throw std::runtime_error(
        "XML config requires <data q=\"...\" v=\"...\" u=\"...\">.");
  }
  if (cfg.solver.horizon == 0) {
    throw std::runtime_error("XML config requires solver horizon > 0.");
  }
  if (cfg.solver.down_sample == 0) {
    throw std::runtime_error("XML config requires solver down_sample > 0.");
  }
  if (!(cfg.solver.interval > 0.)) {
    throw std::runtime_error("XML config requires solver interval > 0.");
  }
}

inline pinocchio::Model load_floating_base_model(
    const crocoddyl::ContactIDRobotConfig& robot) {
  // Contact-ID experiments currently assume a free-flyer base. Robot-specific
  // files provide URDF/SRDF paths through XML instead of hard-coded C++ setup.
  pinocchio::Model model;
  pinocchio::urdf::buildModel(robot.urdf, pinocchio::JointModelFreeFlyer(),
                              model);
  if (!robot.srdf.empty()) {
    pinocchio::srdf::loadReferenceConfigurations(model, robot.srdf, false);
  }
  return model;
}

inline pinocchio::JointIndex resolve_identified_joint(
    const pinocchio::Model& model, const std::string& name) {
  // Users may name either a Pinocchio joint or a link/frame. Frame names are
  // mapped back to their parent joint, which is where inertial parameters live.
  if (model.existJointName(name)) {
    return model.getJointId(name);
  }
  for (pinocchio::FrameIndex i = 0; i < model.frames.size(); ++i) {
    if (model.frames[i].name == name) {
      return model.frames[i].parent;
    }
  }
  throw std::runtime_error("Identified link/joint not found in model: " + name);
}

inline std::vector<pinocchio::JointIndex> resolve_identified_joints(
    const pinocchio::Model& model, const crocoddyl::ContactIDXMLConfig& cfg) {
  std::vector<pinocchio::JointIndex> joints = cfg.identified_joint_indices;
  for (std::size_t i = 0; i < cfg.identified_joint_names.size(); ++i) {
    joints.push_back(resolve_identified_joint(model,
                                              cfg.identified_joint_names[i]));
  }
  if (joints.empty()) {
    throw std::runtime_error(
        "XML config must identify at least one <identification><link>.");
  }
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (joints[i] == 0 ||
        joints[i] >= static_cast<pinocchio::JointIndex>(model.njoints)) {
      throw std::runtime_error("Identified joint index is outside the model.");
    }
  }
  return joints;
}

}  // namespace contact_id_experiment

#endif  // CROCODDYL_EXPERIMENTS_COMMON_CONTACT_ID_MODEL_HPP_
