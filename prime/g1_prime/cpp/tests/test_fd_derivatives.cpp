// Checkpoint A gate 5: centered finite differences of the contact step against
// the analytic dv_dq / dv_dv / dv_dtau across step sizes.  dv_dv and dv_dtau
// come from clean stationarity terms and must be tight; dv_dq is allowed the
// released source omits offset/regularizer mixed terms; the production
// motion-only path includes them analytically and is checked here.
#include "test_common.hpp"

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "g1cal/motion_simulator.hpp"

namespace
{

double rel_err(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
    return (a - b).lpNorm<Eigen::Infinity>() /
           std::max(1.0, b.lpNorm<Eigen::Infinity>());
}

} // namespace

int main()
{
    auto model = g1_test::load_official_model();
    const auto frames = g1_test::official_contact_frames();
    const double dt = 0.01;
    const int nv = model.nv;

    g1cal::MotionAnitescuSimulator::Options opts;
    opts.mu = 1.0;
    opts.plane_height = 0.0;
    opts.kappa = 1000.0;

    std::mt19937 rng(37);
    const std::vector<double> steps = {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6};

    double worst_q = 0., worst_v = 0., worst_tau = 0.;

    for (int trial = 0; trial < 3; ++trial)
    {
        Eigen::VectorXd q, v, tau;
        g1_test::sample_state(model, rng, q, v, tau);

        g1cal::MotionAnitescuSimulator sim(model, frames, opts);
        auto base = sim.step(q, v, tau, dt);
        CHECK(base.diag.newton_converged);

        double best_q = 1e9, best_v = 1e9, best_tau = 1e9;
        for (double h : steps)
        {
            Eigen::MatrixXd fd_q(nv, nv), fd_v(nv, nv), fd_tau(nv, nv);
            for (int i = 0; i < nv; ++i)
            {
                Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
                dq[i] = h;
                Eigen::VectorXd qp(model.nq), qm(model.nq);
                pinocchio::integrate(model, q, dq, qp);
                dq[i] = -h;
                pinocchio::integrate(model, q, dq, qm);
                g1cal::MotionAnitescuSimulator sp(model, frames, opts);
                g1cal::MotionAnitescuSimulator sm(model, frames, opts);
                fd_q.col(i) =
                    (sp.step(qp, v, tau, dt).v_next - sm.step(qm, v, tau, dt).v_next) /
                    (2 * h);

                Eigen::VectorXd vp = v, vm = v;
                vp[i] += h;
                vm[i] -= h;
                g1cal::MotionAnitescuSimulator sv1(model, frames, opts);
                g1cal::MotionAnitescuSimulator sv2(model, frames, opts);
                fd_v.col(i) =
                    (sv1.step(q, vp, tau, dt).v_next - sv2.step(q, vm, tau, dt).v_next) /
                    (2 * h);

                Eigen::VectorXd tp = tau, tm = tau;
                tp[i] += h;
                tm[i] -= h;
                g1cal::MotionAnitescuSimulator st1(model, frames, opts);
                g1cal::MotionAnitescuSimulator st2(model, frames, opts);
                fd_tau.col(i) =
                    (st1.step(q, v, tp, dt).v_next - st2.step(q, v, tm, dt).v_next) /
                    (2 * h);
            }
            best_q = std::min(best_q, rel_err(base.dv_dq, fd_q));
            best_v = std::min(best_v, rel_err(base.dv_dv, fd_v));
            best_tau = std::min(best_tau, rel_err(base.dv_dtau, fd_tau));
            std::cout << "trial " << trial << " h=" << h
                      << " rel_q=" << rel_err(base.dv_dq, fd_q)
                      << " rel_v=" << rel_err(base.dv_dv, fd_v)
                      << " rel_tau=" << rel_err(base.dv_dtau, fd_tau) << "\n";
        }
        std::cout << "trial " << trial << " rel_err best: dv_dq=" << best_q
                  << " dv_dv=" << best_v << " dv_dtau=" << best_tau << "\n";
        worst_q = std::max(worst_q, best_q);
        worst_v = std::max(worst_v, best_v);
        worst_tau = std::max(worst_tau, best_tau);
    }

    // The sweep exposes the inner Newton solve's ~1e-8 stationarity floor;
    // perturbations below that scale amplify solve noise.  These fixed gates
    // cover the best stable step from the full printed sweep on the shipped
    // aligned G1 model.
    CHECK(worst_v < 5e-6);
    CHECK(worst_tau < 5e-6);
    CHECK(worst_q < 1e-4);

    std::cout << "test_fd_derivatives OK (worst dv_dq=" << worst_q
              << ", dv_dv=" << worst_v << ", dv_dtau=" << worst_tau << ")\n";
    return 0;
}
