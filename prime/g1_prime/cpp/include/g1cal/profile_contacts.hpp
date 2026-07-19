// Project-owned derived contact frames for the frozen MuJoCo-GT profile.
//
// The copied data URDF deliberately contains no PRIME-specific eight-corner
// frames. We keep that authority byte-for-byte frozen and add operational
// frames to the in-memory Pinocchio model. The four points per foot are the
// bottom points of the two outer longitudinal MJCF foot capsules.
#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <pinocchio/multibody/model.hpp>

namespace g1cal
{

inline void add_mujoco_profile_contact_frames(
    pinocchio::Model &model, const std::vector<std::string> &requested)
{
    bool missing = false;
    for (const auto &name : requested)
        missing = missing || !model.existFrame(name);
    if (!missing)
        return;

    struct ContactSpec
    {
        const char *name;
        const char *parent_frame;
        Eigen::Vector3d offset;
    };
    const std::array<ContactSpec, 8> specs{{
        {"LL_FOOT_FL", "left_ankle_roll_link", {0.130, 0.010, -0.035}},
        {"LL_FOOT_FR", "left_ankle_roll_link", {0.130, -0.010, -0.035}},
        {"LL_FOOT_RL", "left_ankle_roll_link", {-0.052, 0.010, -0.035}},
        {"LL_FOOT_RR", "left_ankle_roll_link", {-0.052, -0.010, -0.035}},
        {"LR_FOOT_FL", "right_ankle_roll_link", {0.130, 0.010, -0.035}},
        {"LR_FOOT_FR", "right_ankle_roll_link", {0.130, -0.010, -0.035}},
        {"LR_FOOT_RL", "right_ankle_roll_link", {-0.052, 0.010, -0.035}},
        {"LR_FOOT_RR", "right_ankle_roll_link", {-0.052, -0.010, -0.035}},
    }};

    for (const auto &spec : specs)
    {
        if (model.existFrame(spec.name))
            continue;
        if (!model.existFrame(spec.parent_frame))
            throw std::runtime_error(
                std::string("cannot derive contact frame; missing parent ") +
                spec.parent_frame);
        const pinocchio::FrameIndex parent_frame =
            model.getFrameId(spec.parent_frame);
        const auto &parent = model.frames[parent_frame];
        const pinocchio::SE3 placement =
            parent.placement * pinocchio::SE3(Eigen::Matrix3d::Identity(),
                                               spec.offset);
        model.addFrame(pinocchio::Frame(
            spec.name, parent.parentJoint, parent_frame, placement,
            pinocchio::FrameType::OP_FRAME), false);
    }

    for (const auto &name : requested)
        if (!model.existFrame(name))
            throw std::runtime_error("requested contact frame is unavailable: " +
                                     name);
}

} // namespace g1cal
