// Value-path parity: MotionAnitescuSimulator (overlay copy with diagnostics)
// must reproduce the frozen DifferentiableAnitescuSimulator bitwise on
// v_next/derivatives/forces for random interior states.  This is the license
// for using the overlay copy as the motion-only value path.
#include "test_common.hpp"

#include "crocoddyl/contact_id/anitescu/DifferentiableAnitescuSimulator.hpp"
#include "g1cal/motion_simulator.hpp"

int main()
{
    auto model = g1_test::load_official_model();
    const auto frames = g1_test::official_contact_frames();
    const double dt = 0.01;

    DifferentiableAnitescuSimulator::Options fopts;
    fopts.mu = 1.0;
    fopts.plane_height = 0.0;
    fopts.kappa = 1000.0;

    g1cal::MotionAnitescuSimulator::Options mopts;
    mopts.mu = 1.0;
    mopts.plane_height = 0.0;
    mopts.kappa = 1000.0;
    mopts.exact_q_sensitivity = false;
    mopts.newton_max_iters = 100;
    mopts.robust_newton_refinement = false;

    std::mt19937 rng(11);
    int checked = 0;
    for (int trial = 0; trial < 20; ++trial)
    {
        Eigen::VectorXd q, v, tau;
        g1_test::sample_state(model, rng, q, v, tau);

        // Fresh simulators per trial, mirroring the per-calc construction in
        // the frozen action.
        DifferentiableAnitescuSimulator fsim(model, frames, fopts);
        fsim.joints_stack_.clear();
        g1cal::MotionAnitescuSimulator msim(model, frames, mopts);

        StepResult fout = fsim.step(q, v, tau, dt);
        g1cal::MotionStepResult mout = msim.step(q, v, tau, dt);

        // E1 certification assembles the same barrier problem without
        // invoking Newton, then evaluates the accepted candidate directly.
        g1cal::MotionAnitescuSimulator direct(model, frames, mopts);
        direct.prepareProblem(q, v, tau, dt);
        const BarrierSOCP &solved_problem = msim.problem();
        const BarrierSOCP &direct_problem = direct.problem();
        CHECK((solved_problem.H - direct_problem.H)
                  .lpNorm<Eigen::Infinity>() == 0.0);
        CHECK((solved_problem.g - direct_problem.g)
                  .lpNorm<Eigen::Infinity>() == 0.0);
        CHECK(solved_problem.m() == direct_problem.m());
        Eigen::VectorXd solved_grad, direct_grad;
        solved_problem.grad(mout.v_next, solved_grad);
        direct_problem.grad(mout.v_next, direct_grad);
        CHECK((solved_grad - direct_grad).lpNorm<Eigen::Infinity>() == 0.0);
        const double direct_scale = std::max(
            1.0, (direct_problem.H * mout.v_next).norm() +
                     direct_problem.g.norm());
        CHECK(direct_grad.norm() / direct_scale ==
              mout.diag.newton_relative_grad_norm);
        const auto refined = direct.refinePreparedProblem(&mout.v_next);
        Eigen::VectorXd refined_grad;
        direct.problem().grad(refined.v_next, refined_grad);
        const double refined_scale = std::max(
            1.0, (direct.problem().H * refined.v_next).norm() +
                     direct.problem().g.norm());
        CHECK(refined.v_next.allFinite());
        CHECK(refined.force.allFinite());
        CHECK(refined_grad.norm() / refined_scale < 1e-8);
        CHECK(refined.diag.min_cone_margin > 0.0);
        CHECK(refined.diag.min_alpha > 0.0);

        CHECK((fout.v_next - mout.v_next).lpNorm<Eigen::Infinity>() == 0.0);
        CHECK((fout.dv_dq - mout.dv_dq).lpNorm<Eigen::Infinity>() == 0.0);
        CHECK((fout.dv_dv - mout.dv_dv).lpNorm<Eigen::Infinity>() == 0.0);
        CHECK((fout.dv_dtau - mout.dv_dtau).lpNorm<Eigen::Infinity>() == 0.0);
        CHECK((fout.force - mout.force).lpNorm<Eigen::Infinity>() == 0.0);
        CHECK(fout.dv_dtheta.cols() == 0); // frozen sim with empty joint stack

        // The frozen solver discards its boolean return.  Report the raw
        // status without redefining it; value/derivative bitwise parity above
        // proves that a false return is also frozen-path behavior.
        std::cout << "trial=" << trial
                  << " converged=" << mout.diag.newton_converged
                  << " iterations=" << mout.diag.newton_iterations
                  << " grad=" << mout.diag.newton_grad_norm
                  << " rel_grad=" << mout.diag.newton_relative_grad_norm
                  << " feasible_init=" << mout.diag.feasible_init_used
                  << " min_s=" << mout.diag.min_cone_margin
                  << " min_alpha=" << mout.diag.min_alpha << "\n";
        CHECK(std::isfinite(mout.diag.newton_grad_norm));
        CHECK(mout.diag.newton_relative_grad_norm < 1e-8);
        CHECK(mout.diag.min_cone_margin > 0.0);
        CHECK(mout.diag.min_alpha > 0.0);
        ++checked;
    }

    std::cout << "test_sim_parity OK (" << checked << " states, bitwise)\n";
    return 0;
}
