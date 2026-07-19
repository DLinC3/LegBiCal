///////////////////////////////////////////////////////////////////////////////
// g1cal motion-only overlay.
//
// Motion-only data preparation.  Reuses the frozen contact_id_experiment
// helpers (CSV row access, joint reorder, torso->base conversion, control
// assembly, ground shift) and replaces only the state fill: rows are pure
// [q(36), v(35)] with no inertial-parameter slots.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#include "contact_id_preprocess.hpp"

namespace g1cal
{

struct PreparedMotionData
{
    Eigen::VectorXd x0; // nq + nv
    std::vector<Eigen::VectorXd> state_task;
    std::vector<Eigen::VectorXd> ctrl_task;
    std::vector<Eigen::VectorXd> state_init;
    std::vector<Eigen::VectorXd> ctrl_init;
};

inline void fill_motion_state(const pinocchio::Model &model,
                              const crocoddyl::ContactIDDataConfig &data_cfg,
                              const Eigen::MatrixXd &q_csv,
                              const Eigen::MatrixXd &v_csv, std::size_t row,
                              Eigen::VectorXd &x)
{
    x.segment(0, model.nq) = contact_id_experiment::csv_row(
        q_csv, row, data_cfg.q_has_time_column, model.nq);
    if (data_cfg.normalize_base_quaternion)
    {
        contact_id_experiment::normalize_quaternion(x, model.nq);
    }
    x.segment(model.nq, model.nv) = contact_id_experiment::csv_row(
        v_csv, row, data_cfg.v_has_time_column, model.nv);
}

inline PreparedMotionData prepare_motion_data(
    const pinocchio::Model &model,
    const std::vector<std::string> &contact_frames,
    const crocoddyl::ContactIDXMLConfig &cfg)
{
    using namespace contact_id_experiment;

    PreparedMotionData prepared;

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
        last_state_row > static_cast<std::size_t>(v_processed.rows()))
    {
        throw std::runtime_error("State CSVs are shorter than requested horizon.");
    }

    pinocchio::Data pin_data(model);
    const Eigen::Index nx = model.nq + model.nv;

    prepared.x0 = Eigen::VectorXd::Zero(nx);
    fill_motion_state(model, cfg.data, q_processed, v_processed,
                      cfg.solver.start_idx, prepared.x0);

    double initial_ground_height = 0.;
    if (cfg.data.shift_base_to_ground)
    {
        initial_ground_height = lowest_contact_height(
            model, pin_data, contact_frames, prepared.x0.head(model.nq));
        prepared.x0[2] -= initial_ground_height;
    }

    prepared.state_init.push_back(prepared.x0);
    // The motion arrival control is the 70-dim tangent displacement.
    prepared.ctrl_init.push_back(Eigen::VectorXd::Zero(2 * model.nv));

    for (std::size_t i = 0; i < cfg.solver.horizon; i += cfg.solver.down_sample)
    {
        const std::size_t row = cfg.solver.start_idx + i;
        Eigen::VectorXd x_i = Eigen::VectorXd::Zero(nx);
        fill_motion_state(model, cfg.data, q_processed, v_processed, row, x_i);
        if (cfg.data.shift_base_to_ground)
        {
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
         i += cfg.solver.down_sample)
    {
        const std::size_t row = cfg.solver.start_idx + i;
        Eigen::VectorXd u_i = make_control(
            u_processed, row, cfg.solver.down_sample,
            cfg.data.u_has_time_column, model.nv, cfg.data.average_controls);
        prepared.ctrl_task.push_back(u_i);
        prepared.ctrl_init.push_back(u_i);
    }

    return prepared;
}

} // namespace g1cal
