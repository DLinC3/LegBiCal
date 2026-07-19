// Shared helpers for the g1cal overlay tests (shipped G1 model).
#pragma once

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "g1cal/profile_contacts.hpp"

#define CHECK(cond)                                                        \
    do                                                                     \
    {                                                                      \
        if (!(cond))                                                       \
        {                                                                  \
            std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__ \
                      << ": " #cond << "\n";                               \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                 \
    do                                                                        \
    {                                                                         \
        const double _a = (a), _b = (b);                                      \
        if (!(std::abs(_a - _b) <= (tol)))                                    \
        {                                                                     \
            std::cerr << "CHECK_NEAR failed at " << __FILE__ << ":"          \
                      << __LINE__ << ": " << _a << " vs " << _b              \
                      << " tol=" << (tol) << "\n";                            \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

namespace g1_test
{

inline std::string project_root()
{
    const char *env = std::getenv("G1CAL_ROOT");
    if (env != nullptr)
        return std::string(env);
#ifdef G1CAL_SOURCE_ROOT
    return std::string(G1CAL_SOURCE_ROOT);
#else
    return std::string(".");
#endif
}

inline std::string official_urdf()
{
    return project_root() +
           "/models/g1/urdf/g1_custom_collision_29dof.urdf";
}

inline std::vector<std::string> official_contact_frames()
{
    return {"LL_FOOT_FL", "LL_FOOT_FR", "LL_FOOT_RL", "LL_FOOT_RR",
            "LR_FOOT_FL", "LR_FOOT_FR", "LR_FOOT_RL", "LR_FOOT_RR"};
}

inline pinocchio::Model load_official_model()
{
    pinocchio::Model model;
    pinocchio::urdf::buildModel(official_urdf(), pinocchio::JointModelFreeFlyer(),
                                model);
    g1cal::add_mujoco_profile_contact_frames(model, official_contact_frames());
    return model;
}

inline double lowest_contact_z(const pinocchio::Model &model,
                               const Eigen::VectorXd &q)
{
    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    double min_z = std::numeric_limits<double>::infinity();
    for (const auto &name : official_contact_frames())
    {
        const auto fid = model.getFrameId(name);
        min_z = std::min(min_z, data.oMf[fid].translation().z());
    }
    return min_z;
}

// Deterministic near-standing interior state with contact activity: after
// perturbation the configuration is shifted so the lowest foot corner sits
// slightly above the ground (Phi_min = +1 mm), mirroring the official
// shift_base_to_ground preprocessing while staying strictly cone-interior.
inline void sample_state(const pinocchio::Model &model, std::mt19937 &rng,
                         Eigen::VectorXd &q, Eigen::VectorXd &v,
                         Eigen::VectorXd &tau, double joint_amp = 0.05,
                         double vel_amp = 0.2, double tau_amp = 5.0)
{
    std::normal_distribution<double> N(0., 1.);
    q = model.referenceConfigurations.count("standing")
            ? model.referenceConfigurations.at("standing")
            : pinocchio::neutral(model);
    for (int i = 7; i < model.nq; ++i)
        q[i] += joint_amp * N(rng);
    Eigen::Vector4d quat = q.segment<4>(3);
    quat += 0.02 * Eigen::Vector4d(N(rng), N(rng), N(rng), N(rng));
    quat.normalize();
    q.segment<4>(3) = quat;

    q[2] -= lowest_contact_z(model, q) - 1e-3;

    v = Eigen::VectorXd::Zero(model.nv);
    for (int i = 0; i < model.nv; ++i)
        v[i] = vel_amp * N(rng);

    tau = Eigen::VectorXd::Zero(model.nv);
    for (int i = 6; i < model.nv; ++i)
        tau[i] = tau_amp * N(rng);
}

} // namespace g1_test
