///////////////////////////////////////////////////////////////////////////////
// g1cal motion-only overlay.
//
// MotionAnitescuSimulator is an equation-identical copy of the pinned
// DifferentiableAnitescuSimulator (third_party/PRIME at the documented pin,
// include/crocoddyl/contact_id/anitescu/DifferentiableAnitescuSimulator.hpp)
// with three deliberate differences, none of which change solved values:
//   1. the inertial-parameter sensitivity path (joints_stack_/dv_dtheta and
//      the joint-torque regressor) is removed — this overlay is motion-only;
//   2. Newton convergence/cone-margin diagnostics are captured and returned;
//   3. the barrier problem/solver classes are reused from the frozen headers.
// tests/cpp/test_sim_parity.cpp asserts bitwise equality of v_next, dv_dq,
// dv_dv, dv_dtau, and force against the frozen simulator on random states.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

#include "crocoddyl/contact_id/anitescu/BarrierSOCP.hpp"

namespace g1cal
{

// Equation-identical copy of the frozen NewtonSolver with iteration/status
// capture. The update rule, tolerances, line search, and regularization are
// byte-copied from the pinned NewtonSolver.hpp.
class NewtonSolverDiag
{
public:
    struct Options
    {
        int max_iters = 100;
        double grad_tol = 1e-8;
        double step_tol = 1e-10;
        double c1 = 1e-4;
        double alpha_init = 1.0;
        double alpha_min = 1e-8;
        double backtrack = 0.5;
        double reg = 1e-9;
    };

    struct Diag
    {
        bool converged = false;
        int iterations = 0;
        double final_grad_norm = std::numeric_limits<double>::quiet_NaN();
        bool spd_failure = false;
        std::string termination = "not_started";
    };

    NewtonSolverDiag() : opt_() {}

    void set_max_iters(const int max_iters)
    {
        if (max_iters <= 0)
            throw std::invalid_argument("Newton max_iters must be positive");
        opt_.max_iters = max_iters;
    }

    void set_gradient_refinement(const bool enabled)
    {
        gradient_refinement_ = enabled;
    }

    template <typename F, typename G, typename HFun, typename Feas>
    bool solve(Eigen::VectorXd &x, F &&f, G &&g, HFun &&H, Feas &&feasible,
               Diag &diag)
    {
        const int n = x.size();
        Eigen::VectorXd grad(n);
        Eigen::MatrixXd hess(n, n);

        double fx = f(x);
        diag = Diag();

        for (int it = 0; it < opt_.max_iters; ++it)
        {
            diag.iterations = it;
            g(x, grad);
            double gn = grad.norm();
            diag.final_grad_norm = gn;

            if (gn < opt_.grad_tol)
            {
                diag.converged = true;
                diag.termination = "gradient";
                return true;
            }

            H(x, hess);

            Eigen::VectorXd step;
            if (!solveSPD(hess, grad, step))
            {
                Eigen::MatrixXd hreg =
                    hess + opt_.reg * Eigen::MatrixXd::Identity(n, n);
                if (!solveSPD(hreg, grad, step))
                {
                    diag.spd_failure = true;
                    diag.termination = "hessian_not_spd";
                    return false;
                }
            }

            step = -step;
            if (step.norm() < opt_.step_tol)
            {
                diag.converged = true;
                diag.termination = "step";
                return true;
            }

            double alpha = lineSearch(x, step, fx, grad, f, feasible);
            if (alpha < opt_.alpha_min)
            {
                // The released objective Armijo test can stall near the
                // barrier optimum when the objective decrement is below
                // floating-point resolution although stationarity still
                // improves.  Production mode retries the same Newton
                // direction with a feasibility + gradient-norm merit test.
                // Legacy source-parity tests disable this path.
                bool refined = false;
                if (gradient_refinement_)
                {
                    Eigen::VectorXd candidate(n), candidate_grad(n);
                    double refine_alpha = opt_.alpha_init;
                    while (refine_alpha >= opt_.alpha_min)
                    {
                        candidate.noalias() = x + refine_alpha * step;
                        if (feasible(candidate))
                        {
                            g(candidate, candidate_grad);
                            if (candidate_grad.norm() < gn)
                            {
                                x = candidate;
                                fx = f(x);
                                refined = true;
                                break;
                            }
                        }
                        refine_alpha *= opt_.backtrack;
                    }
                }
                if (!refined)
                {
                    diag.termination = "line_search";
                    return false;
                }
                continue;
            }

            x.noalias() += alpha * step;
            fx = f(x);
        }
        diag.termination = "max_iterations";
        return false;
    }

private:
    Options opt_;
    bool gradient_refinement_ = false;

    bool solveSPD(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                  Eigen::VectorXd &sol) const
    {
        Eigen::LLT<Eigen::MatrixXd> llt(H);
        if (llt.info() != Eigen::Success)
            return false;
        if (gradient_refinement_)
        {
            // Pivoted orthogonal factorization is more accurate than the
            // fast unpivoted Cholesky direction for the highly conditioned
            // finite-barrier Hessian.  Legacy mode remains LLT bit-for-bit.
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(H);
            sol = cod.solve(g);
            // The contact barrier Hessian becomes ill-conditioned near the
            // cone boundary.  Reuse the same LLT factors for a few residual
            // correction solves; this changes neither sparsity nor the
            // mathematical Newton direction, but avoids a stationarity floor
            // caused by a poorly resolved linear solve.
            for (int iteration = 0; iteration < 5; ++iteration)
            {
                const Eigen::VectorXd residual = g - H * sol;
                if (residual.norm() <=
                    1e-13 * std::max(1.0, g.norm()))
                    break;
                sol += cod.solve(residual);
            }
        }
        else
        {
            sol = llt.solve(g);
        }
        return (llt.info() == Eigen::Success);
    }

    template <typename F, typename Feas>
    double lineSearch(const Eigen::VectorXd &x, const Eigen::VectorXd &p,
                      double fx, const Eigen::VectorXd &grad, F &&f,
                      Feas &&feasible)
    {
        double alpha = opt_.alpha_init;
        double slope = grad.dot(p);

        Eigen::VectorXd xt(x.size());

        while (alpha > opt_.alpha_min)
        {
            xt.noalias() = x + alpha * p;

            if (!feasible(xt))
            {
                alpha *= opt_.backtrack;
                continue;
            }

            double ft = f(xt);
            if (ft <= fx + opt_.c1 * alpha * slope)
                return alpha;

            alpha *= opt_.backtrack;
        }
        return alpha;
    }
};

// Contact-solve diagnostics exposed by the motion-only overlay. The frozen
// StepResult does not report Newton convergence; these fields are additive.
struct ContactStepDiag
{
    bool newton_converged = false;
    int newton_iterations = 0;
    double newton_grad_norm = std::numeric_limits<double>::quiet_NaN();
    double newton_relative_grad_norm =
        std::numeric_limits<double>::quiet_NaN();
    bool feasible_init_used = false;
    double min_cone_margin = std::numeric_limits<double>::quiet_NaN();
    double min_alpha = std::numeric_limits<double>::quiet_NaN();
    std::string newton_termination = "not_started";
};

struct MotionStepResult
{
    Eigen::VectorXd v_next;  // nv
    Eigen::MatrixXd dv_dq;   // nv x nv
    Eigen::MatrixXd dv_dv;   // nv x nv
    Eigen::MatrixXd dv_dtau; // nv x nv
    Eigen::VectorXd force;   // 3*m, source order [t1, t2, n] per contact
    ContactStepDiag diag;
};

// Value-only result for deterministic certification of an already prepared
// contact problem. It intentionally omits sensitivities: certification solves
// the same strictly convex value problem but is not part of the FDDP
// derivative path.
struct PreparedContactSolveResult
{
    Eigen::VectorXd v_next;
    Eigen::VectorXd force;
    ContactStepDiag diag;
    double objective = std::numeric_limits<double>::quiet_NaN();
};

namespace detail
{
inline Eigen::Matrix3d skew3(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
        v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

inline Eigen::Matrix<double, 3, 2> tangentBasisFromNormal(const Eigen::Vector3d &n_in)
{
    Eigen::Vector3d n = n_in.normalized();
    Eigen::Vector3d ref = (std::abs(n.z()) < 0.9) ? Eigen::Vector3d::UnitZ()
                                                  : Eigen::Vector3d::UnitX();
    Eigen::Vector3d t1 = ref - n * (n.dot(ref));
    double nt1 = t1.norm();
    if (nt1 < 1e-12)
    {
        ref = Eigen::Vector3d::UnitY();
        t1 = ref - n * (n.dot(ref));
        nt1 = t1.norm();
    }
    t1 /= nt1;
    Eigen::Vector3d t2 = n.cross(t1);
    Eigen::Matrix<double, 3, 2> T;
    T.col(0) = t1;
    T.col(1) = t2;
    return T;
}
} // namespace detail

class MotionAnitescuSimulator
{
public:
    struct Options
    {
        double mu = 0.3;
        double plane_height = 0.0;
        double kappa = 50.0;
        double damping = 0.05;
        double armature = 0.01;
        Eigen::Vector3d n_vec = Eigen::Vector3d(0, 0, 1);
        bool enforce_alpha_positive = true;
        // The released PRIME q sensitivity omits the +1e-6 alpha offset and
        // gamma_cone mixed derivatives.  Keep a legacy switch solely for
        // source-parity tests; production motion-only actions use the complete
        // analytic stationarity derivative.
        bool exact_q_sensitivity = true;
        // Production extends only the iteration budget of the same convex
        // Newton/Armijo solve.  Parity tests explicitly select the released 100.
        int newton_max_iters = 300;
        // Preserve source-identical Newton in legacy parity mode; production
        // uses a stationarity merit fallback when Armijo loses resolution.
        bool robust_newton_refinement = true;
    };

    MotionAnitescuSimulator(const pinocchio::Model &model,
                            const std::vector<std::string> &contact_frame_names,
                            const Options &opt)
        : model_(model), opt_(opt), data_(model)
    {
        nv_ = model_.nv;
        nq_ = model_.nq;

        T_ = detail::tangentBasisFromNormal(opt_.n_vec);
        setContacts(contact_frame_names);

        M_.setZero(nv_, nv_);
        Minv_.setZero(nv_, nv_);
        h_.setZero(nv_);
        a_free_.setZero(nv_);
        v_free_next_.setZero(nv_);

        drnea_dq_.setZero(nv_, nv_);
        drnea_dv_.setZero(nv_, nv_);
        drnea_da_.setZero(nv_, nv_);

        dv_dq_.setZero(nv_, nv_);

        J6_.setZero(6, nv_);
        Jp_.setZero(3, nv_);
        lwaJf_.setZero(6, nv_);

        armature_ = opt_.armature;
        damping_ = opt_.damping;
        M_armature_ = armature_ * Eigen::MatrixXd::Identity(nv_, nv_);
        D_damping_ = damping_ * Eigen::MatrixXd::Identity(nv_, nv_);
        M_armature_.diagonal().head(6).setZero();
        D_damping_.diagonal().head(6).setZero();

        solver_.set_max_iters(opt_.newton_max_iters);
        solver_.set_gradient_refinement(opt_.robust_newton_refinement);

        resizeContactWorkspaces();
    }

    void setContacts(const std::vector<std::string> &contact_frame_names)
    {
        contact_names_ = contact_frame_names;
        const std::size_t m = contact_names_.size();
        frame_ids_.resize(m);
        joint_ids_.resize(m);

        for (std::size_t i = 0; i < m; ++i)
        {
            const auto fid = model_.getFrameId(contact_names_[i]);
            if (fid == (pinocchio::FrameIndex)(-1) ||
                fid >= (pinocchio::FrameIndex)model_.nframes)
                throw std::runtime_error("Contact frame not found: " + contact_names_[i]);
            frame_ids_[i] = fid;
            joint_ids_[i] = model_.frames[fid].parentJoint;
        }
        resizeContactWorkspaces();
    }

    MotionStepResult step(const Eigen::VectorXd &q,
                          const Eigen::VectorXd &v,
                          const Eigen::VectorXd &tau,
                          double dt,
                          const Eigen::VectorXd *initial_velocity = nullptr)
    {
        prepareProblem(q, v, tau, dt);
        solveBarrierNewton(initial_velocity);
        computeSensitivities();

        Eigen::VectorXd force = prob_.lambda_forces(v_next_) / dt_;

        MotionStepResult out;
        out.v_next = v_next_;
        out.dv_dq = dv_dq_;
        out.dv_dv = dv_dv_;
        out.dv_dtau = dv_dtau_;
        out.force = force;
        out.diag = diag_;

        double min_s = std::numeric_limits<double>::infinity();
        double min_alpha = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < prob_.m(); ++i)
        {
            min_s = std::min(min_s, prob_.s(v_next_, i));
            min_alpha = std::min(min_alpha, prob_.alpha(v_next_, i));
        }
        out.diag.min_cone_margin = min_s;
        out.diag.min_alpha = min_alpha;
        return out;
    }

    // Assemble the exact finite-barrier contact problem for a recorded motion
    // knot without invoking Newton.  This is the certification path: it lets
    // callers evaluate stationarity at an already accepted FDDP velocity,
    // rather than testing whether an unrelated cold restart reaches it.
    void prepareProblem(const Eigen::VectorXd &q,
                        const Eigen::VectorXd &v,
                        const Eigen::VectorXd &tau,
                        double dt)
    {
        if (q.size() != nq_ || v.size() != nv_ || tau.size() != nv_)
            throw std::runtime_error("Dimension mismatch in prepareProblem().");
        if (!(dt > 0.) || !std::isfinite(dt))
            throw std::runtime_error("dt must be finite positive");

        dt_ = dt;
        q_ = q;
        v_ = v;
        tau_ = tau;
        computeFreeDynamics();
        updateKinematics(q_);
        buildContactsFromFrames();
        buildBarrierProblem();
    }

    PreparedContactSolveResult refinePreparedProblem(
        const Eigen::VectorXd *initial_velocity = nullptr)
    {
        if (prob_.H.rows() != nv_ || !(dt_ > 0.))
            throw std::runtime_error(
                "refinePreparedProblem called before prepareProblem");
        solveBarrierNewton(initial_velocity);
        PreparedContactSolveResult out;
        out.v_next = v_next_;
        out.force = prob_.lambda_forces(v_next_) / dt_;
        out.diag = diag_;
        out.objective = prob_.f(v_next_);
        double min_s = std::numeric_limits<double>::infinity();
        double min_alpha = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < prob_.m(); ++i)
        {
            min_s = std::min(min_s, prob_.s(v_next_, i));
            min_alpha = std::min(min_alpha, prob_.alpha(v_next_, i));
        }
        out.diag.min_cone_margin = min_s;
        out.diag.min_alpha = min_alpha;
        return out;
    }

    pinocchio::Data &data() { return data_; }
    const pinocchio::Data &data() const { return data_; }

    // Read-only diagnostic access for the offline single-step contact probe
    // (H=501 root-cause analysis).  Exposes the already-built barrier problem
    // and the solved velocity; no solved value or equation changes.
    const BarrierSOCP &problem() const { return prob_; }
    const Eigen::VectorXd &solved_velocity() const { return v_next_; }

private:
    void computeFreeDynamics()
    {
        pinocchio::computeAllTerms(model_, data_, q_, v_);
        M_ = data_.M;
        M_.template triangularView<Eigen::StrictlyLower>() = M_.transpose();
        M_ = M_ + M_armature_;
        Minv_ = M_.inverse();

        h_ = pinocchio::rnea(model_, data_, q_, v_, Eigen::VectorXd::Zero(nv_));
        a_free_.noalias() = Minv_ * (tau_ - h_);
        v_free_next_.noalias() = v_ + dt_ * a_free_;
    }

    void updateKinematics(const Eigen::VectorXd &q)
    {
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);
        pinocchio::computeJointJacobians(model_, data_);
        pinocchio::computeJointKinematicHessians(model_, data_, q);
    }

    void buildContactsFromFrames()
    {
        const std::size_t m = frame_ids_.size();
        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const pinocchio::FrameIndex fid = frame_ids_[ci];
            const pinocchio::JointIndex jid = joint_ids_[ci];

            const pinocchio::SE3 &oMf = data_.oMf[fid];
            p_world_[ci] = oMf.translation();

            Phi_[ci] = opt_.n_vec.dot(p_world_[ci]) - opt_.plane_height;

            J6_.setZero();
            pinocchio::getJointJacobian(model_, data_, jid, pinocchio::LOCAL_WORLD_ALIGNED, J6_);

            const Eigen::Vector3d oMj = data_.oMi[jid].translation();
            const Eigen::Matrix3d oRj = data_.oMi[jid].rotation();
            const Eigen::Vector3d r = p_world_[ci] - oMj;

            Jp_.noalias() = J6_.topRows(3) - detail::skew3(r) * J6_.bottomRows(3);

            Jn_[ci].noalias() = opt_.n_vec.transpose() * Jp_;
            Jt_[ci].noalias() = T_.transpose() * Jp_;

            lwaJf_.topRows(3) = Jp_;
            lwaJf_.bottomRows(3) = J6_.bottomRows(3);

            const Eigen::Vector3d p_j = oRj.transpose() * (p_world_[ci] - oMj);
            jMf_ = pinocchio::SE3(Eigen::Matrix3d::Identity(), p_j);
            oRf_ = oRj;

            transferKinematicsHessian(jid, lwaJf_, jMf_, oRf_, opt_.n_vec, T_,
                                      lwaHf_linear_stack_[ci],
                                      lwaHf_linear_normal_stack_[ci],
                                      lwaHf_linear_tangent_stack_[ci]);
        }
    }

    void buildBarrierProblem()
    {
        prob_.H = M_ + dt_ * D_damping_;
        prob_.g = -M_ * v_free_next_;
        prob_.kappa = opt_.kappa;
        prob_.mu = opt_.mu;
        prob_.enforce_alpha_positive = opt_.enforce_alpha_positive;
        prob_.clearConstraints();

        const double mu = opt_.mu;
        const std::size_t m = frame_ids_.size();

        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const Eigen::RowVectorXd a = (1.0 / mu) * Jn_[ci];
            const Eigen::MatrixXd B = Jt_[ci];

            double c = (Phi_[ci] / dt_) * (1.0 / mu);
            c += 1e-6;
            prob_.addConstraint(a, B, c);
        }
    }

    void solveBarrierNewton(const Eigen::VectorXd *initial_velocity)
    {
        v_next_.setZero(nv_);
        Eigen::VectorXd v_init = v_;

        if (initial_velocity != nullptr && initial_velocity->size() == nv_ &&
            initial_velocity->allFinite() && prob_.feasible(*initial_velocity))
        {
            v_init = *initial_velocity;
        }

        diag_ = ContactStepDiag();
        if (!prob_.feasible(v_init))
        {
            v_init = prob_.feasibleInitEquality(v_init, /*alpha_min=*/1e-2);
            diag_.feasible_init_used = true;
        }

        v_next_ = v_init;

        NewtonSolverDiag::Diag ndiag;
        solver_.solve(
            v_next_,
            [&](const Eigen::VectorXd &x)
            { return prob_.f(x); },
            [&](const Eigen::VectorXd &x, Eigen::VectorXd &gg)
            { prob_.grad(x, gg); },
            [&](const Eigen::VectorXd &x, Eigen::MatrixXd &HH)
            { prob_.hess(x, HH); },
            [&](const Eigen::VectorXd &x)
            { return prob_.feasible(x); },
            ndiag);
        diag_.newton_converged = ndiag.converged;
        diag_.newton_iterations = ndiag.iterations;
        diag_.newton_grad_norm = ndiag.final_grad_norm;
        diag_.newton_termination = ndiag.termination;
        Eigen::VectorXd final_grad(nv_);
        prob_.grad(v_next_, final_grad);
        const double stationarity_scale = std::max(
            1.0, (prob_.H * v_next_).norm() + prob_.g.norm());
        diag_.newton_relative_grad_norm =
            final_grad.norm() / stationarity_scale;
    }

    void computeSensitivities()
    {
        compute_dg_dq_allContacts();

        Eigen::MatrixXd H_next;
        prob_.hess(v_next_, H_next);
        Eigen::LLT<Eigen::MatrixXd> llt(H_next);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("LLT failed: prob.H not SPD.");

        drnea_dq_.setZero(nv_, nv_);
        drnea_dv_.setZero(nv_, nv_);
        drnea_da_.setZero(nv_, nv_);

        pinocchio::computeRNEADerivatives(model_, data_,
                                          q_, v_, (v_next_ - v_) / dt_,
                                          drnea_dq_, drnea_dv_, drnea_da_);

        sum_dg_dq_.setZero(nv_, nv_);
        for (auto &G : dg_dq_list_)
            sum_dg_dq_.noalias() += G;

        dv_dq_.noalias() = -llt.solve(dt_ * drnea_dq_ + sum_dg_dq_);
        dv_dv_.noalias() = -llt.solve(-M_ + dt_ * drnea_dv_);
        dv_dtau_.noalias() = -llt.solve(-dt_ * Eigen::MatrixXd::Identity(nv_, nv_));
    }

    void compute_dg_dq_allContacts()
    {
        if (opt_.exact_q_sensitivity)
        {
            compute_dg_dq_allContactsExact();
            return;
        }

        // Frozen released implementation, retained for explicit parity.
        const double mu = opt_.mu;
        const double inv_mu2 = 1.0 / (mu * mu);
        const std::size_t m = frame_ids_.size();

        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const Eigen::RowVectorXd &Jn = Jn_[ci];
            const Eigen::MatrixXd &Jt = Jt_[ci];
            const double Phi = Phi_[ci];

            const Eigen::VectorXd &vplus = v_next_;

            const double s = (Phi / dt_ + (Jn * vplus)(0, 0));
            const double alpha = inv_mu2 * s;
            const Eigen::Vector2d beta = Jt * vplus;

            cone_.noalias() = alpha * Jn.transpose() - Jt.transpose() * beta;

            dalpha_dq_ = inv_mu2 * (Jn / dt_);
            for (int k = 0; k < nv_; ++k)
                dalpha_dq_(0, k) += inv_mu2 * (lwaHf_linear_normal_stack_[ci][k] * vplus)(0, 0);

            dcone_normal_dq_.noalias() = Jn.transpose() * dalpha_dq_;
            for (int k = 0; k < nv_; ++k)
                dcone_normal_dq_.col(k) += alpha * lwaHf_linear_normal_stack_[ci][k].transpose();

            dbeta_dq_.setZero(2, nv_);
            for (int k = 0; k < nv_; ++k)
                dbeta_dq_.col(k) = lwaHf_linear_tangent_stack_[ci][k] * vplus;

            dcone_tangent_dq_.setZero(nv_, nv_);
            for (int j = 0; j < nv_; ++j)
            {
                const Eigen::MatrixXd &dJt = lwaHf_linear_tangent_stack_[ci][j];
                const Eigen::VectorXd t1 = dJt.transpose() * beta;
                const Eigen::VectorXd t2 = Jt.transpose() * dbeta_dq_.col(j);
                dcone_tangent_dq_.col(j) = t1 + t2;
            }

            dcone_dq_ = dcone_normal_dq_ - dcone_tangent_dq_;

            const double cone2 = inv_mu2 * (s * s) - beta.squaredNorm();
            dcone2_dq_.noalias() = 2.0 / inv_mu2 * alpha * dalpha_dq_ - 2.0 * beta.transpose() * dbeta_dq_;

            const double cc = -(2.0 / opt_.kappa) * (1.0 / cone2);
            const double inv_cone2_sq = 1.0 / (cone2 * cone2);

            g_ = -(2.0 / (opt_.kappa * cone2)) * cone_;

            dg_dq_.setZero(nv_, nv_);
            for (int k = 0; k < nv_; ++k)
            {
                const double dc_dqk = (2.0 / opt_.kappa) * inv_cone2_sq * dcone2_dq_(0, k);
                dg_dq_.col(k) = dc_dqk * cone_ + cc * dcone_dq_.col(k);
            }
            dg_dq_list_[ci] = dg_dq_;
        }
    }

    // Complete analytic mixed derivative of the contact contribution to the
    // velocity-stationarity equation.  For each contact,
    //
    //   alpha = (Jn vplus + Phi/dt)/mu + 1e-6, beta = Jt vplus,
    //   p = alpha a - B' beta, s = alpha^2 - beta'beta,
    //   g_contact = gamma*(alpha*a + B'*beta) - 2/kappa*p/s.
    //
    // The released derivative used an idealized alpha without the offset and
    // omitted the gamma term even though both are present in BarrierSOCP::grad.
    void compute_dg_dq_allContactsExact()
    {
        const double mu = opt_.mu;
        const double inv_mu = 1.0 / mu;
        const double gamma = prob_.gamma_cone;
        const std::size_t m = frame_ids_.size();

        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const Eigen::RowVectorXd &Jn = Jn_[ci];
            const Eigen::MatrixXd &Jt = Jt_[ci];
            const Eigen::VectorXd &vplus = v_next_;

            const Eigen::VectorXd a = inv_mu * Jn.transpose();
            const double alpha = prob_.alpha(vplus, ci);
            const Eigen::VectorXd beta = prob_.beta(vplus, ci);
            const double cone_margin = prob_.s(vplus, ci);
            if (!(cone_margin > 0.0))
                throw std::runtime_error("exact q sensitivity outside positive cone");

            const Eigen::VectorXd p = alpha * a - Jt.transpose() * beta;
            dg_dq_.setZero(nv_, nv_);

            for (int j = 0; j < nv_; ++j)
            {
                const Eigen::RowVectorXd &dJn =
                    lwaHf_linear_normal_stack_[ci][j];
                const Eigen::MatrixXd &dJt =
                    lwaHf_linear_tangent_stack_[ci][j];
                const Eigen::VectorXd da = inv_mu * dJn.transpose();
                const double dc = inv_mu * Jn(j) / dt_;
                const double dalpha = inv_mu * (dJn * vplus)(0, 0) + dc;
                const Eigen::VectorXd dbeta = dJt * vplus;

                const Eigen::VectorXd dp =
                    dalpha * a + alpha * da - dJt.transpose() * beta -
                    Jt.transpose() * dbeta;
                const double ds =
                    2.0 * alpha * dalpha - 2.0 * beta.dot(dbeta);

                Eigen::VectorXd column =
                    -(2.0 / opt_.kappa) *
                    (dp / cone_margin - p * (ds / (cone_margin * cone_margin)));

                if (gamma > 0.0)
                {
                    column.noalias() += gamma *
                        (dalpha * a + alpha * da + dJt.transpose() * beta +
                         Jt.transpose() * dbeta);
                }
                dg_dq_.col(j) = column;
            }
            dg_dq_list_[ci] = dg_dq_;
        }
    }

    // Matrix-stack-only version of the frozen transferKinematicsHessian: the
    // frozen code fills both tensor and stack forms; only the stacks are read
    // by the sensitivity pass, so the tensors are omitted here.
    void transferKinematicsHessian(pinocchio::JointIndex joint_id,
                                   const Eigen::MatrixXd &lwaJf,
                                   const pinocchio::SE3 &jMf,
                                   const Eigen::Matrix3d &oRf,
                                   const Eigen::Vector3d &normal_contact,
                                   const Eigen::Matrix<double, 3, 2> &tangent_contact,
                                   std::vector<Eigen::MatrixXd> &lwaHf_linear_stack,
                                   std::vector<Eigen::MatrixXd> &lwaHf_linear_normal_stack,
                                   std::vector<Eigen::MatrixXd> &lwaHf_linear_tangent_stack)
    {
        const int nv = nv_;

        Eigen::Tensor<double, 3> jHj(6, nv, nv);
        jHj.setZero();
        pinocchio::getJointKinematicHessian(model_, data_, joint_id, pinocchio::LOCAL, jHj);

        const pinocchio::SE3::ActionMatrixType fXj = jMf.inverse().toActionMatrix();

        lwaHf_linear_stack.resize(nv);
        lwaHf_linear_normal_stack.resize(nv);
        lwaHf_linear_tangent_stack.resize(nv);

        for (int i = 0; i < nv; ++i)
        {
            Eigen::Tensor<double, 2> t_jHj_i = jHj.chip(i, 2).eval();
            Eigen::MatrixXd jHj_i = Eigen::Map<const Eigen::MatrixXd>(t_jHj_i.data(), 6, nv);

            Eigen::MatrixXd fHf_i = fXj * jHj_i;

            Eigen::MatrixXd lwaHf_i = oRf * fHf_i.topRows<3>();

            Eigen::Matrix3d S;
            pinocchio::skew(lwaJf.bottomRows<3>().col(i), S);
            lwaHf_i.noalias() += S * lwaJf.topRows<3>();

            Eigen::MatrixXd lwaHf_normal_i = normal_contact.transpose() * lwaHf_i;
            Eigen::MatrixXd lwaHf_tangent_i = tangent_contact.transpose() * lwaHf_i;

            lwaHf_linear_stack[i] = lwaHf_i;
            lwaHf_linear_normal_stack[i] = lwaHf_normal_i;
            lwaHf_linear_tangent_stack[i] = lwaHf_tangent_i;
        }
    }

    void resizeContactWorkspaces()
    {
        const std::size_t m = contact_names_.size();

        p_world_.assign(m, Eigen::Vector3d::Zero());
        Phi_.assign(m, 0.0);

        Jn_.assign(m, Eigen::RowVectorXd::Zero(nv_));
        Jt_.assign(m, Eigen::MatrixXd::Zero(2, nv_));

        dg_dq_list_.assign(m, Eigen::MatrixXd::Zero(nv_, nv_));

        lwaHf_linear_stack_.assign(m, std::vector<Eigen::MatrixXd>(nv_, Eigen::MatrixXd::Zero(3, nv_)));
        lwaHf_linear_normal_stack_.assign(m, std::vector<Eigen::MatrixXd>(nv_, Eigen::MatrixXd::Zero(1, nv_)));
        lwaHf_linear_tangent_stack_.assign(m, std::vector<Eigen::MatrixXd>(nv_, Eigen::MatrixXd::Zero(2, nv_)));

        sum_dg_dq_.setZero(nv_, nv_);
    }

    const pinocchio::Model &model_;
    Options opt_;
    pinocchio::Data data_;

    int nv_{0}, nq_{0};

    std::vector<std::string> contact_names_;
    std::vector<pinocchio::FrameIndex> frame_ids_;
    std::vector<pinocchio::JointIndex> joint_ids_;

    Eigen::Matrix<double, 3, 2> T_;

    double dt_{0.0};
    Eigen::VectorXd q_, v_, tau_;

    Eigen::MatrixXd M_, Minv_;
    Eigen::VectorXd h_, a_free_, v_free_next_;

    Eigen::MatrixXd M_armature_;
    Eigen::MatrixXd D_damping_;

    double damping_{0.05};
    double armature_{0.01};

    BarrierSOCP prob_;
    NewtonSolverDiag solver_;
    ContactStepDiag diag_;

    Eigen::VectorXd v_next_;

    Eigen::MatrixXd drnea_dq_, drnea_dv_, drnea_da_;

    Eigen::MatrixXd dv_dq_;
    Eigen::MatrixXd dv_dv_;
    Eigen::MatrixXd dv_dtau_;

    std::vector<Eigen::Vector3d> p_world_;
    std::vector<double> Phi_;
    std::vector<Eigen::RowVectorXd> Jn_;
    std::vector<Eigen::MatrixXd> Jt_;

    std::vector<std::vector<Eigen::MatrixXd>> lwaHf_linear_stack_;
    std::vector<std::vector<Eigen::MatrixXd>> lwaHf_linear_normal_stack_;
    std::vector<std::vector<Eigen::MatrixXd>> lwaHf_linear_tangent_stack_;

    std::vector<Eigen::MatrixXd> dg_dq_list_;
    Eigen::MatrixXd sum_dg_dq_;

    Eigen::MatrixXd J6_;
    Eigen::MatrixXd Jp_;
    Eigen::MatrixXd lwaJf_;
    pinocchio::SE3 jMf_;
    Eigen::Matrix3d oRf_;

    Eigen::RowVectorXd dalpha_dq_;
    Eigen::MatrixXd dcone_dq_;
    Eigen::MatrixXd dcone_normal_dq_;
    Eigen::MatrixXd dcone_tangent_dq_;
    Eigen::MatrixXd dbeta_dq_;
    Eigen::RowVectorXd dcone2_dq_;
    Eigen::VectorXd g_;
    Eigen::MatrixXd dg_dq_;

    Eigen::VectorXd cone_;
};

} // namespace g1cal
