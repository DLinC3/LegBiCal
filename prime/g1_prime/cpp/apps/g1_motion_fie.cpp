///////////////////////////////////////////////////////////////////////////////
// g1cal motion-only overlay.
//
// g1_motion_fie: fixed-inertia motion-only FIE runner.  Consumes the same XML
// configuration format as the official runners (the <identification> section
// is ignored: there are no identified links in this estimator) and writes
// trajectory CSVs plus a JSON diagnostics summary.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"

#include "contact_id_model.hpp"
#include "contact_id_outputs.hpp"

#include "g1cal/motion_preprocess.hpp"
#include "g1cal/motion_problem.hpp"
#include "g1cal/profile_contacts.hpp"

namespace
{

std::vector<double> make_solver_alphas(double alpha0)
{
    std::vector<double> alphas;
    for (int i = 0; i < 11; ++i)
    {
        alphas.push_back(std::ldexp(alpha0, -i));
    }
    return alphas;
}

void validate_motion_config(const crocoddyl::ContactIDXMLConfig &cfg)
{
    if (cfg.robot.urdf.empty())
        throw std::runtime_error("XML config requires <robot urdf>");
    if (cfg.contact_frames.empty())
        throw std::runtime_error("XML config requires contact frames");
    if (cfg.data.q_csv.empty() || cfg.data.v_csv.empty() || cfg.data.u_csv.empty())
        throw std::runtime_error("XML config requires q/v/u csv");
    if (cfg.solver.horizon == 0 || cfg.solver.down_sample == 0 ||
        !(cfg.solver.interval > 0.))
        throw std::runtime_error("XML solver config invalid");
}

void save_sequence(const std::string &path,
                   const std::vector<Eigen::VectorXd> &seq)
{
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f.is_open())
        throw std::runtime_error("cannot open " + path);
    f << std::setprecision(17);
    for (const auto &v : seq)
    {
        for (Eigen::Index i = 0; i < v.size(); ++i)
        {
            f << v[i];
            if (i + 1 < v.size())
                f << ",";
        }
        f << "\n";
    }
}

void save_sequence_atomic(const std::string &path,
                          const std::vector<Eigen::VectorXd> &seq)
{
    const std::string temporary = path + ".tmp";
    save_sequence(temporary, seq);
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot atomically replace " + path);
}

std::vector<Eigen::VectorXd> load_sequence(const std::string &path,
                                           const Eigen::Index expected_cols)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("cannot open " + path);
    std::vector<Eigen::VectorXd> rows;
    std::string line, field;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::vector<double> values;
        while (std::getline(ss, field, ','))
            if (!field.empty())
                values.push_back(std::stod(field));
        if (static_cast<Eigen::Index>(values.size()) != expected_cols)
            throw std::runtime_error("wrong column count in " + path);
        Eigen::VectorXd row(expected_cols);
        for (Eigen::Index i = 0; i < expected_cols; ++i)
            row[i] = values[static_cast<std::size_t>(i)];
        rows.push_back(row);
    }
    return rows;
}

std::vector<Eigen::VectorXd> load_controls(const std::string &path,
                                           const Eigen::Index arrival_cols,
                                           const Eigen::Index running_cols)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("cannot open " + path);
    std::vector<Eigen::VectorXd> rows;
    std::string line, field;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::vector<double> values;
        while (std::getline(ss, field, ','))
            if (!field.empty())
                values.push_back(std::stod(field));
        const Eigen::Index expected = rows.empty() ? arrival_cols : running_cols;
        if (static_cast<Eigen::Index>(values.size()) != expected)
            throw std::runtime_error("wrong control column count in " + path);
        Eigen::VectorXd row(expected);
        for (Eigen::Index i = 0; i < expected; ++i)
            row[i] = values[static_cast<std::size_t>(i)];
        rows.push_back(row);
    }
    return rows;
}

struct ContactCandidateMetrics
{
    std::string seed_mode;
    std::string termination;
    int iterations = 0;
    bool feasible_init_used = false;
    Eigen::VectorXd velocity;
    Eigen::VectorXd force;
    bool feasible = false;
    bool llt_success = false;
    bool stationarity_passed = false;
    double objective = std::numeric_limits<double>::quiet_NaN();
    double grad_norm = std::numeric_limits<double>::quiet_NaN();
    double relative_grad = std::numeric_limits<double>::quiet_NaN();
    double min_margin = std::numeric_limits<double>::infinity();
    double min_alpha = std::numeric_limits<double>::infinity();
    double eig_min = std::numeric_limits<double>::quiet_NaN();
    double eig_max = std::numeric_limits<double>::quiet_NaN();
    double condition = std::numeric_limits<double>::infinity();
    double decrement = std::numeric_limits<double>::quiet_NaN();
    double step_norm = std::numeric_limits<double>::quiet_NaN();
};

ContactCandidateMetrics evaluate_candidate(
    const BarrierSOCP &problem, const Eigen::VectorXd &velocity,
    const double dt, const std::string &seed_mode,
    const std::string &termination, const int iterations,
    const bool feasible_init_used)
{
    constexpr double tolerance = 1e-7;
    ContactCandidateMetrics out;
    out.seed_mode = seed_mode;
    out.termination = termination;
    out.iterations = iterations;
    out.feasible_init_used = feasible_init_used;
    out.velocity = velocity;
    out.feasible = problem.feasible(velocity);
    if (!out.feasible)
        return out;

    Eigen::VectorXd gradient(velocity.size());
    Eigen::MatrixXd hessian;
    problem.grad(velocity, gradient);
    problem.hess(velocity, hessian);
    out.objective = problem.f(velocity);
    out.grad_norm = gradient.norm();
    const double scale = std::max(
        1.0, (problem.H * velocity).norm() + problem.g.norm());
    out.relative_grad = out.grad_norm / scale;
    for (std::size_t c = 0; c < problem.m(); ++c)
    {
        out.min_margin = std::min(out.min_margin, problem.s(velocity, c));
        out.min_alpha = std::min(out.min_alpha, problem.alpha(velocity, c));
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(hessian);
    if (eigensolver.info() == Eigen::Success)
    {
        out.eig_min = eigensolver.eigenvalues().minCoeff();
        out.eig_max = eigensolver.eigenvalues().maxCoeff();
        if (out.eig_min > 0.)
            out.condition = out.eig_max / out.eig_min;
    }
    Eigen::LLT<Eigen::MatrixXd> llt(hessian);
    out.llt_success = llt.info() == Eigen::Success;
    if (out.llt_success)
    {
        const Eigen::VectorXd step = -llt.solve(gradient);
        out.step_norm = step.norm();
        out.decrement = std::sqrt(std::max(0., -gradient.dot(step)));
    }
    out.force = problem.lambda_forces(velocity) / dt;
    out.stationarity_passed =
        std::isfinite(out.relative_grad) && out.relative_grad <= tolerance &&
        out.min_margin > 0. && out.min_alpha > 0. && out.llt_success;
    return out;
}

struct CertificationSummary
{
    double defect_max = 0.;
    double defect_sum = 0.;
    double inner_grad_max = 0.;
    double inner_relative_grad_max = 0.;
    double inner_min_cone_margin = std::numeric_limits<double>::infinity();
    double inner_min_alpha = std::numeric_limits<double>::infinity();
    double inner_hessian_condition_max = 0.;
    int inner_stationarity_rejected = 0;
    double accepted_node_relative_grad_max = 0.;
    int accepted_node_stationarity_rejected = 0;
    double action_velocity_mismatch_max = 0.;
    int cold_restart_not_converged = 0;
    int cold_restart_stationarity_rejected = 0;
    int cold_restart_feasible_init_used = 0;
    int cold_restart_iter_max = 0;
    double cold_restart_relative_grad_max = 0.;
    int selected_accepted_direct = 0;
    int selected_accepted_refined = 0;
    int selected_cold_refined = 0;
    std::size_t n_models = 0;
    std::size_t n_contact_knots = 0;
};

CertificationSummary certify_accepted_trajectory(
    const crocoddyl::ContactIDXMLConfig &cfg,
    const pinocchio::Model &model,
    g1cal::MotionFIEProblem &builder,
    const boost::shared_ptr<crocoddyl::ShootingProblem> &shooting,
    const std::vector<Eigen::VectorXd> &xs,
    const std::vector<Eigen::VectorXd> &us,
    const std::string &output_dir)
{
    constexpr double tolerance = 1e-7;
    const auto &models = shooting->get_runningModels();
    const auto &datas = shooting->get_runningDatas();
    auto state = builder.get_state();
    if (xs.size() != models.size() + 1 || us.size() != models.size())
        throw std::runtime_error("certification trajectory horizon mismatch");

    contact_id_experiment::ensure_dir(output_dir);
    g1cal::MotionAnitescuSimulator::Options options;
    options.mu = cfg.weights.mu;
    options.kappa = cfg.weights.kappa;
    options.exact_q_sensitivity = true;
    options.newton_max_iters = 1000;
    options.robust_newton_refinement = true;
    g1cal::MotionAnitescuSimulator certifier(
        model, cfg.contact_frames, options);
    const double dt = cfg.solver.down_sample * cfg.solver.interval;
    const int nq = model.nq;
    const int nv = model.nv;

    std::ofstream contact_file(output_dir + "/contact_diagnostics.csv");
    std::ofstream candidate_file(
        output_dir + "/contact_candidate_diagnostics.csv");
    std::ofstream corner_file(output_dir + "/contact_corner_diagnostics.csv");
    if (!contact_file || !candidate_file || !corner_file)
        throw std::runtime_error("cannot open certification diagnostics output");
    contact_file << std::setprecision(17);
    candidate_file << std::setprecision(17);
    corner_file << std::setprecision(17);
    contact_file
        << "knot,newton_converged,newton_termination,newton_iterations,"
           "newton_grad_norm,newton_relative_grad_norm,feasible_init_used,"
           "min_cone_margin,min_alpha,force_norm,certification_mode,"
           "selected_seed_mode,selected_objective,selected_grad_norm,"
           "selected_relative_grad_norm,selected_stationarity_passed,"
           "selected_min_cone_margin,selected_min_alpha,hessian_llt_success,"
           "hessian_eig_min,hessian_eig_max,hessian_condition,"
           "newton_decrement,newton_step_norm,action_velocity_mismatch,"
           "accepted_node_feasible,accepted_node_grad_norm,"
           "accepted_node_relative_grad_norm,accepted_node_stationarity_passed,"
           "cold_restart_converged,cold_restart_termination,"
           "cold_restart_iterations,cold_restart_grad_norm,"
           "cold_restart_relative_grad_norm,cold_restart_feasible_init_used,"
           "cold_restart_min_cone_margin,cold_restart_min_alpha,"
           "cold_restart_force_norm\n";
    candidate_file
        << "knot,candidate_index,seed_mode,termination,iterations,"
           "feasible_init_used,feasible,objective,grad_norm,relative_grad_norm,"
           "stationarity_passed,min_cone_margin,min_alpha,hessian_llt_success,"
           "hessian_eig_min,hessian_eig_max,hessian_condition,"
           "newton_decrement,newton_step_norm,force_norm\n";
    corner_file
        << "knot,contact_index,contact_frame,alpha,beta_t1,beta_t2,"
           "cone_margin,normal_velocity_coordinate,impulse_t1,impulse_t2,"
           "impulse_normal,force_t1,force_t2,force_normal,force_norm\n";

    CertificationSummary summary;
    summary.n_models = models.size();
    Eigen::VectorXd defect(state->get_ndx());
    std::vector<Eigen::VectorXd> certified_forces;
    for (std::size_t k = 0; k < models.size(); ++k)
    {
        if (k == 0)
        {
            models[k]->calc(datas[k], xs[k], us[k]);
            state->diff(datas[k]->xnext, xs[k + 1], defect);
        }
        else
        {
            const Eigen::VectorXd q = xs[k].head(nq);
            const Eigen::VectorXd v = xs[k].tail(nv);
            const Eigen::VectorXd accepted_v = xs[k + 1].tail(nv);
            certifier.prepareProblem(q, v, us[k], dt);
            const BarrierSOCP &problem = certifier.problem();

            std::vector<ContactCandidateMetrics> candidates;
            candidates.push_back(evaluate_candidate(
                problem, accepted_v, dt, "accepted_direct", "direct", 0,
                false));
            const auto accepted_refined =
                certifier.refinePreparedProblem(&accepted_v);
            candidates.push_back(evaluate_candidate(
                problem, accepted_refined.v_next, dt, "accepted_refined",
                accepted_refined.diag.newton_termination,
                accepted_refined.diag.newton_iterations,
                accepted_refined.diag.feasible_init_used));
            const auto cold_refined = certifier.refinePreparedProblem(nullptr);
            candidates.push_back(evaluate_candidate(
                problem, cold_refined.v_next, dt, "cold_refined",
                cold_refined.diag.newton_termination,
                cold_refined.diag.newton_iterations,
                cold_refined.diag.feasible_init_used));

            auto valid = [](const ContactCandidateMetrics &candidate)
            {
                return candidate.feasible && candidate.llt_success &&
                       std::isfinite(candidate.relative_grad);
            };
            auto better = [&](const ContactCandidateMetrics &a,
                              const ContactCandidateMetrics &b)
            {
                if (valid(a) != valid(b))
                    return valid(a);
                if (!valid(a))
                    return false;
                if (a.relative_grad != b.relative_grad)
                    return a.relative_grad < b.relative_grad;
                if (a.objective != b.objective)
                    return a.objective < b.objective;
                return a.seed_mode < b.seed_mode;
            };
            std::size_t selected_index = 0;
            for (std::size_t i = 1; i < candidates.size(); ++i)
                if (better(candidates[i], candidates[selected_index]))
                    selected_index = i;
            const ContactCandidateMetrics &selected = candidates[selected_index];
            const ContactCandidateMetrics &accepted_node = candidates[0];
            const ContactCandidateMetrics &cold = candidates[2];

            for (std::size_t i = 0; i < candidates.size(); ++i)
            {
                const auto &candidate = candidates[i];
                candidate_file
                    << (k - 1) << "," << i << "," << candidate.seed_mode
                    << "," << candidate.termination << ","
                    << candidate.iterations << ","
                    << (candidate.feasible_init_used ? 1 : 0) << ","
                    << (candidate.feasible ? 1 : 0) << ","
                    << candidate.objective << "," << candidate.grad_norm << ","
                    << candidate.relative_grad << ","
                    << (candidate.stationarity_passed ? 1 : 0) << ","
                    << candidate.min_margin << "," << candidate.min_alpha << ","
                    << (candidate.llt_success ? 1 : 0) << ","
                    << candidate.eig_min << "," << candidate.eig_max << ","
                    << candidate.condition << "," << candidate.decrement << ","
                    << candidate.step_norm << "," << candidate.force.norm()
                    << "\n";
            }

            if (selected.seed_mode == "accepted_direct")
                ++summary.selected_accepted_direct;
            else if (selected.seed_mode == "accepted_refined")
                ++summary.selected_accepted_refined;
            else
                ++summary.selected_cold_refined;
            if (!selected.stationarity_passed)
                ++summary.inner_stationarity_rejected;
            if (!accepted_node.stationarity_passed)
                ++summary.accepted_node_stationarity_rejected;
            if (!cold_refined.diag.newton_converged)
                ++summary.cold_restart_not_converged;
            if (!cold.stationarity_passed)
                ++summary.cold_restart_stationarity_rejected;
            if (cold_refined.diag.feasible_init_used)
                ++summary.cold_restart_feasible_init_used;
            summary.cold_restart_iter_max = std::max(
                summary.cold_restart_iter_max,
                cold_refined.diag.newton_iterations);
            summary.cold_restart_relative_grad_max = std::max(
                summary.cold_restart_relative_grad_max, cold.relative_grad);
            summary.inner_grad_max =
                std::max(summary.inner_grad_max, selected.grad_norm);
            summary.inner_relative_grad_max = std::max(
                summary.inner_relative_grad_max, selected.relative_grad);
            summary.inner_min_cone_margin = std::min(
                summary.inner_min_cone_margin, selected.min_margin);
            summary.inner_min_alpha =
                std::min(summary.inner_min_alpha, selected.min_alpha);
            if (std::isfinite(selected.condition))
                summary.inner_hessian_condition_max = std::max(
                    summary.inner_hessian_condition_max, selected.condition);
            if (std::isfinite(accepted_node.relative_grad))
                summary.accepted_node_relative_grad_max = std::max(
                    summary.accepted_node_relative_grad_max,
                    accepted_node.relative_grad);
            const double velocity_mismatch =
                (selected.velocity - accepted_v).norm();
            summary.action_velocity_mismatch_max = std::max(
                summary.action_velocity_mismatch_max, velocity_mismatch);

            Eigen::VectorXd certified_dx =
                Eigen::VectorXd::Zero(state->get_ndx());
            certified_dx.head(nv) = dt * selected.velocity;
            certified_dx.tail(nv) = selected.velocity - v;
            Eigen::VectorXd certified_xnext(state->get_nx());
            state->integrate(xs[k], certified_dx, certified_xnext);
            state->diff(certified_xnext, xs[k + 1], defect);
            certified_forces.push_back(selected.force);
            ++summary.n_contact_knots;

            for (std::size_t c = 0; c < problem.m(); ++c)
            {
                const Eigen::VectorXd beta = problem.beta(selected.velocity, c);
                const Eigen::Vector3d impulse =
                    selected.force.segment<3>(3 * c) * dt;
                const Eigen::Vector3d force =
                    selected.force.segment<3>(3 * c);
                corner_file
                    << (k - 1) << "," << c << "," << cfg.contact_frames[c]
                    << "," << problem.alpha(selected.velocity, c) << ","
                    << beta[0] << "," << beta[1] << ","
                    << problem.s(selected.velocity, c) << ","
                    << cfg.weights.mu * problem.alpha(selected.velocity, c)
                    << "," << impulse[0] << "," << impulse[1] << ","
                    << impulse[2] << "," << force[0] << "," << force[1]
                    << "," << force[2] << "," << force.norm() << "\n";
            }

            contact_file
                << (k - 1) << ","
                << (selected.stationarity_passed ? 1 : 0) << ","
                << selected.termination << "," << selected.iterations << ","
                << selected.grad_norm << "," << selected.relative_grad << ","
                << (selected.feasible_init_used ? 1 : 0) << ","
                << selected.min_margin << "," << selected.min_alpha << ","
                << selected.force.norm() << ","
                << "action_stationarity_plus_shooting_defect_v3,"
                << selected.seed_mode << "," << selected.objective << ","
                << selected.grad_norm << "," << selected.relative_grad << ","
                << (selected.stationarity_passed ? 1 : 0) << ","
                << selected.min_margin << "," << selected.min_alpha << ","
                << (selected.llt_success ? 1 : 0) << "," << selected.eig_min
                << "," << selected.eig_max << "," << selected.condition << ","
                << selected.decrement << "," << selected.step_norm << ","
                << velocity_mismatch << ","
                << (accepted_node.feasible ? 1 : 0) << ","
                << accepted_node.grad_norm << ","
                << accepted_node.relative_grad << ","
                << (accepted_node.stationarity_passed ? 1 : 0) << ","
                << (cold_refined.diag.newton_converged ? 1 : 0) << ","
                << cold.termination << "," << cold.iterations << ","
                << cold.grad_norm << "," << cold.relative_grad << ","
                << (cold.feasible_init_used ? 1 : 0) << ","
                << cold.min_margin << "," << cold.min_alpha << ","
                << cold.force.norm() << "\n";
        }

        const double defect_norm = defect.norm();
        summary.defect_max = std::max(summary.defect_max, defect_norm);
        summary.defect_sum += defect_norm;
    }

    if (!certified_forces.empty())
        save_sequence(output_dir + "/" + cfg.outputs.force_log,
                      certified_forces);
    return summary;
}

void write_certification_summary(const std::string &path,
                                 const CertificationSummary &s,
                                 const std::string &mode)
{
    std::ofstream js(path, std::ios::out | std::ios::trunc);
    if (!js.is_open())
        throw std::runtime_error("cannot open " + path);
    js << std::setprecision(17)
       << "{\n  \"schema\": \"g1cal_contact_certification_v3\",\n"
       << "  \"mode\": \"" << mode << "\",\n"
       << "  \"contact_certification_mode\": "
          "\"action_stationarity_plus_shooting_defect_v3\",\n"
       << "  \"inner_relative_grad_tolerance\": 1e-7,\n"
       << "  \"inner_refinement_max_iters\": 1000,\n"
       << "  \"defect_tolerance\": 1e-6,\n"
       << "  \"n_running_models\": " << s.n_models << ",\n"
       << "  \"n_contact_knots\": " << s.n_contact_knots << ",\n"
       << "  \"defect_max\": " << s.defect_max << ",\n"
       << "  \"defect_mean\": "
       << (s.n_models ? s.defect_sum / s.n_models : 0.) << ",\n"
       << "  \"inner_grad_max\": " << s.inner_grad_max << ",\n"
       << "  \"inner_relative_grad_max\": "
       << s.inner_relative_grad_max << ",\n"
       << "  \"inner_stationarity_rejected\": "
       << s.inner_stationarity_rejected << ",\n"
       << "  \"contact_health_passed\": "
       << (s.inner_stationarity_rejected == 0 ? "true" : "false") << ",\n"
       << "  \"inner_min_cone_margin\": " << s.inner_min_cone_margin << ",\n"
       << "  \"inner_min_alpha\": " << s.inner_min_alpha << ",\n"
       << "  \"inner_hessian_condition_max\": "
       << s.inner_hessian_condition_max << ",\n"
       << "  \"accepted_node_relative_grad_max\": "
       << s.accepted_node_relative_grad_max << ",\n"
       << "  \"accepted_node_stationarity_rejected\": "
       << s.accepted_node_stationarity_rejected << ",\n"
       << "  \"action_velocity_mismatch_max\": "
       << s.action_velocity_mismatch_max << ",\n"
       << "  \"cold_restart_not_converged\": "
       << s.cold_restart_not_converged << ",\n"
       << "  \"cold_restart_stationarity_rejected\": "
       << s.cold_restart_stationarity_rejected << ",\n"
       << "  \"cold_restart_feasible_init_used\": "
       << s.cold_restart_feasible_init_used << ",\n"
       << "  \"cold_restart_iter_max\": " << s.cold_restart_iter_max << ",\n"
       << "  \"cold_restart_relative_grad_max\": "
       << s.cold_restart_relative_grad_max << ",\n"
       << "  \"selected_seed_counts\": {\"accepted_direct\": "
       << s.selected_accepted_direct << ", \"accepted_refined\": "
       << s.selected_accepted_refined << ", \"cold_refined\": "
       << s.selected_cold_refined << "}\n}\n";
}

long resident_set_kib()
{
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key)
    {
        if (key == "VmRSS:")
        {
            long value = 0;
            std::string unit;
            status >> value >> unit;
            return value;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return -1;
}

class LongRunCallback final : public crocoddyl::CallbackAbstract
{
  public:
    LongRunCallback(const std::string &output_dir,
                    const std::size_t checkpoint_interval)
        : output_dir_(output_dir),
          checkpoint_interval_(checkpoint_interval),
          start_(std::chrono::steady_clock::now())
    {
        contact_id_experiment::ensure_dir(output_dir_);
        contact_id_experiment::ensure_dir(output_dir_ + "/checkpoint");
        progress_.open(output_dir_ + "/progress.jsonl",
                       std::ios::out | std::ios::app);
        if (!progress_.is_open())
            throw std::runtime_error("cannot open long-run progress log");
    }

    void operator()(crocoddyl::SolverAbstract &solver) override
    {
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_).count();
        progress_ << std::setprecision(17)
                  << "{\"iteration\":" << solver.get_iter()
                  << ",\"cost\":" << solver.get_cost()
                  << ",\"stop\":" << solver.get_stop()
                  << ",\"steplength\":" << solver.get_steplength()
                  << ",\"elapsed_seconds\":" << elapsed
                  << ",\"rss_kib\":" << resident_set_kib()
                  << "}\n";
        progress_.flush();

        const std::size_t iteration = solver.get_iter();
        if (checkpoint_interval_ > 0 && iteration > 0 &&
            iteration % checkpoint_interval_ == 0)
        {
            const std::string checkpoint = output_dir_ + "/checkpoint";
            save_sequence_atomic(checkpoint + "/xs_results_fddp.csv",
                                 solver.get_xs());
            save_sequence_atomic(checkpoint + "/us_results_fddp.csv",
                                 solver.get_us());
            std::ofstream metadata(checkpoint + "/checkpoint.json.tmp",
                                   std::ios::out | std::ios::trunc);
            metadata << std::setprecision(17)
                     << "{\n  \"iteration\": " << iteration
                     << ",\n  \"cost\": " << solver.get_cost()
                     << ",\n  \"stop\": " << solver.get_stop()
                     << ",\n  \"elapsed_seconds\": " << elapsed
                     << "\n}\n";
            metadata.close();
            const std::string temporary = checkpoint + "/checkpoint.json.tmp";
            const std::string final = checkpoint + "/checkpoint.json";
            if (std::rename(temporary.c_str(), final.c_str()) != 0)
                throw std::runtime_error("cannot replace checkpoint metadata");
        }
    }

  private:
    std::string output_dir_;
    std::size_t checkpoint_interval_;
    std::chrono::steady_clock::time_point start_;
    std::ofstream progress_;
};

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const bool certify_only =
            argc >= 2 && std::string(argv[1]) == "--certify-only";
        if ((!certify_only &&
             !(argc == 2 || argc == 3 || argc == 4 || argc == 6 || argc == 7)) ||
            (certify_only && argc != 6))
        {
            std::cerr << "Usage: g1_motion_fie <config.xml> "
                         "[precision.csv [warm_xs.csv warm_arrival_u.csv "
                         "warm_running_us.csv] [prior_state.csv]]\n"
                         "       g1_motion_fie --certify-only <config.xml> "
                         "<xs.csv> <us.csv> <output_dir>\n";
            return 2;
        }

        const int config_arg = certify_only ? 2 : 1;
        const crocoddyl::ContactIDXMLConfig cfg =
            crocoddyl::contact_id_xml::load_config(argv[config_arg]);
        validate_motion_config(cfg);

        if (cfg.solver.dry_run)
        {
            std::cout << "Parsed motion-FIE XML successfully.\n";
            std::cout << "contacts=" << cfg.contact_frames.size() << "\n";
            return 0;
        }

        pinocchio::Model model =
            contact_id_experiment::load_floating_base_model(cfg.robot);
        g1cal::add_mujoco_profile_contact_frames(model, cfg.contact_frames);

        g1cal::PreparedMotionData data =
            g1cal::prepare_motion_data(model, cfg.contact_frames, cfg);

        const bool warm_start_used = !certify_only && (argc == 6 || argc == 7);
        const bool prior_override_used = !certify_only && (argc == 4 || argc == 7);
        if (prior_override_used)
        {
            const int prior_arg = argc == 4 ? 3 : 6;
            const auto prior = load_sequence(argv[prior_arg],
                                              model.nq + model.nv);
            if (prior.size() != 1)
                throw std::runtime_error("prior state file must have one row");
            data.x0 = prior.front();
            data.state_init.front() = data.x0;
        }

        g1cal::DiagonalCovariancePrecision covariance;
        if (!certify_only && argc >= 3)
            covariance = g1cal::load_covariance_precision(
                argv[2], 2 * model.nv, model.nv);

        const char *newton_env = std::getenv("G1CAL_NEWTON_MAX_ITERS");
        const int newton_max_iters = newton_env ? std::stoi(newton_env) : 300;
        g1cal::MotionFIEProblem builder(model, cfg.contact_frames,
                                           cfg.weights, covariance,
                                           newton_max_iters);

        const double timestep = cfg.solver.down_sample * cfg.solver.interval;
        auto shooting = builder.createEstimationProblem(
            data.x0, timestep, data.state_task.size(), data.state_task,
            data.ctrl_task);
        shooting->set_nthreads(cfg.solver.n_thread);

        if (certify_only)
        {
            const auto xs = load_sequence(argv[3], model.nq + model.nv);
            const auto us = load_controls(argv[4], 2 * model.nv, model.nv);
            const std::string output_dir = argv[5];
            const CertificationSummary certification =
                certify_accepted_trajectory(cfg, model, builder, shooting,
                                            xs, us, output_dir);
            write_certification_summary(
                output_dir + "/certification_summary.json", certification,
                "certify_only_no_fddp");
            std::cout << "certify_only_ok contacts="
                      << certification.n_contact_knots
                      << " inner_rejected="
                      << certification.inner_stationarity_rejected
                      << " accepted_node_rejected="
                      << certification.accepted_node_stationarity_rejected
                      << " cold_restart_rejected="
                      << certification.cold_restart_stationarity_rejected
                      << "\n";
            return certification.inner_stationarity_rejected == 0 &&
                           certification.defect_max < 1e-6
                       ? 0
                       : 3;
        }

        crocoddyl::SolverFDDP solver(shooting);
        solver.set_alphas(make_solver_alphas(cfg.solver.alpha0));
        double initial_regularization = 0.1;
        if (const char *regularization_env =
                std::getenv("G1CAL_INITIAL_REGULARIZATION"))
        {
            initial_regularization = std::stod(regularization_env);
            if (!(initial_regularization > 0.) ||
                !std::isfinite(initial_regularization))
            {
                throw std::runtime_error(
                    "G1CAL_INITIAL_REGULARIZATION must be finite positive");
            }
            solver.set_preg(initial_regularization);
            solver.set_dreg(initial_regularization);
        }
        const char *checkpoint_env =
            std::getenv("G1CAL_CHECKPOINT_INTERVAL");
        const std::size_t checkpoint_interval = checkpoint_env
            ? static_cast<std::size_t>(std::stoul(checkpoint_env))
            : 0;
        const bool enable_long_run_callback = checkpoint_env != nullptr;
        if (cfg.solver.callbacks || enable_long_run_callback)
        {
            std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
            if (cfg.solver.callbacks)
                cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
            if (enable_long_run_callback)
                cbs.push_back(boost::make_shared<LongRunCallback>(
                    cfg.outputs.directory, checkpoint_interval));
            solver.setCallbacks(cbs);
        }

        std::vector<Eigen::VectorXd> xs_init = data.state_init;
        std::vector<Eigen::VectorXd> us_init = data.ctrl_init;
        if (warm_start_used)
        {
            xs_init = load_sequence(argv[3], model.nq + model.nv);
            const auto arrival_u = load_sequence(argv[4], 2 * model.nv);
            const auto running_us = load_sequence(argv[5], model.nv);
            if (xs_init.size() != data.state_init.size() ||
                arrival_u.size() != 1 ||
                running_us.size() + 1 != data.ctrl_init.size())
                throw std::runtime_error("warm-start horizon mismatch");
            us_init.clear();
            us_init.push_back(arrival_u.front());
            us_init.insert(us_init.end(), running_us.begin(), running_us.end());
        }

        crocoddyl::Timer timer;
        const bool solved = solver.solve(xs_init, us_init,
                                         cfg.solver.max_iter, false,
                                         initial_regularization > 0.
                                             ? initial_regularization
                                             : 0.1);
        const double duration_s = timer.get_duration() / 1000.;
        std::cout << "Duration: " << duration_s << " seconds\n";

        const auto &xs = solver.get_xs();
        const auto &us = solver.get_us();

        contact_id_experiment::ensure_dir(cfg.outputs.directory);
        save_sequence(
            contact_id_experiment::output_path(cfg.outputs, cfg.outputs.xs_results),
            xs);
        save_sequence(
            contact_id_experiment::output_path(cfg.outputs, cfg.outputs.us_results),
            us);

        // V3 separately certifies the unique nested contact action solution
        // and the multiple-shooting manifold defect.
        const CertificationSummary certification =
            certify_accepted_trajectory(cfg, model, builder, shooting, xs, us,
                                        cfg.outputs.directory);
        const auto &models = shooting->get_runningModels();
        auto state = builder.get_state();

        std::ofstream js(contact_id_experiment::output_path(cfg.outputs,
                                                            "solve_summary.json"),
                         std::ios::out | std::ios::trunc);
        js << std::setprecision(17);
        js << "{\n"
           << "  \"solved\": " << (solved ? "true" : "false") << ",\n"
           << "  \"cost_mode\": \""
           << (covariance.enabled ? "strict_covariance" : "legacy_weights")
           << "\",\n"
           << "  \"covariance_config_hash\": \""
           << (covariance.enabled ? covariance.config_hash : "") << "\",\n"
           << "  \"warm_start_used\": "
           << (warm_start_used ? "true" : "false") << ",\n"
           << "  \"prior_override_used\": "
           << (prior_override_used ? "true" : "false") << ",\n"
           << "  \"iterations\": " << solver.get_iter() << ",\n"
           << "  \"initial_regularization\": "
           << initial_regularization << ",\n"
           << "  \"final_preg\": " << solver.get_preg() << ",\n"
           << "  \"final_dreg\": " << solver.get_dreg() << ",\n"
           << "  \"final_cost\": " << solver.get_cost() << ",\n"
           << "  \"stop\": " << solver.get_stop() << ",\n"
           << "  \"steplength\": " << solver.get_steplength() << ",\n"
           << "  \"n_running_models\": " << models.size() << ",\n"
           << "  \"nx\": " << state->get_nx() << ",\n"
           << "  \"ndx\": " << state->get_ndx() << ",\n"
           << "  \"contact_certification_mode\": "
              "\"action_stationarity_plus_shooting_defect_v3\",\n"
           << "  \"defect_max\": " << certification.defect_max << ",\n"
           << "  \"defect_mean\": "
           << (models.empty() ? 0. : certification.defect_sum / models.size())
           << ",\n"
           << "  \"defect_tolerance\": 1e-6,\n"
           << "  \"inner_relative_grad_tolerance\": 1e-7,\n"
           << "  \"inner_refinement_max_iters\": 1000,\n"
           << "  \"inner_grad_max\": "
           << certification.inner_grad_max << ",\n"
           << "  \"inner_relative_grad_max\": "
           << certification.inner_relative_grad_max << ",\n"
           << "  \"inner_stationarity_rejected\": "
           << certification.inner_stationarity_rejected << ",\n"
           << "  \"contact_health_passed\": "
           << (certification.inner_stationarity_rejected == 0 ? "true" : "false")
           << ",\n"
           << "  \"inner_min_cone_margin\": "
           << certification.inner_min_cone_margin << ",\n"
           << "  \"inner_min_alpha\": "
           << certification.inner_min_alpha << ",\n"
           << "  \"inner_hessian_condition_max\": "
           << certification.inner_hessian_condition_max << ",\n"
           << "  \"accepted_node_relative_grad_max\": "
           << certification.accepted_node_relative_grad_max << ",\n"
           << "  \"accepted_node_stationarity_rejected\": "
           << certification.accepted_node_stationarity_rejected << ",\n"
           << "  \"action_velocity_mismatch_max\": "
           << certification.action_velocity_mismatch_max << ",\n"
           << "  \"cold_restart_not_converged\": "
           << certification.cold_restart_not_converged << ",\n"
           << "  \"cold_restart_stationarity_rejected\": "
           << certification.cold_restart_stationarity_rejected << ",\n"
           << "  \"newton_not_converged\": "
           << certification.cold_restart_not_converged << ",\n"
           << "  \"newton_max_iters\": " << newton_max_iters << ",\n"
           << "  \"newton_iter_max\": "
           << certification.cold_restart_iter_max << ",\n"
           << "  \"newton_relative_grad_max\": "
           << certification.inner_relative_grad_max << ",\n"
           << "  \"newton_relative_grad_tolerance\": 1e-7,\n"
           << "  \"newton_stationarity_rejected\": "
           << certification.inner_stationarity_rejected << ",\n"
           << "  \"feasible_init_used\": "
           << certification.cold_restart_feasible_init_used << ",\n"
           << "  \"min_cone_margin\": "
           << certification.inner_min_cone_margin << ",\n"
           << "  \"duration_seconds\": " << duration_s << "\n"
           << "}\n";

        std::cout << "iterations=" << solver.get_iter()
                  << " cost=" << solver.get_cost()
                  << " defect_max=" << certification.defect_max
                  << " inner_rejected="
                  << certification.inner_stationarity_rejected
                  << " accepted_node_rejected="
                  << certification.accepted_node_stationarity_rejected
                  << " cold_restart_rejected="
                  << certification.cold_restart_stationarity_rejected << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "g1_motion_fie: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
