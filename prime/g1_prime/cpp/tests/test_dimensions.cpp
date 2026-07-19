// Checkpoint A gates 1-2: motion-only public dimensions are exactly
// nx=71, ndx=70, running nu=35, arrival nu=70; no parameter slice exists.
#include "test_common.hpp"

#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "g1cal/motion_problem.hpp"

int main()
{
    auto model = g1_test::load_official_model();
    CHECK(model.nq == 36);
    CHECK(model.nv == 35);

    crocoddyl::ContactIDWeights weights;
    weights.kappa = 1000.0;
    weights.mu = 1.0;

    g1cal::MotionFIEProblem builder(model, g1_test::official_contact_frames(),
                                       weights);
    auto state = builder.get_state();
    CHECK(state->get_nx() == 71);
    CHECK(state->get_ndx() == 70);
    CHECK(state->get_nq() == 36);
    CHECK(state->get_nv() == 35);
    CHECK(state->get_nq_pin() == 36);
    CHECK(state->get_nv_pin() == 35);

    // Build a 3-knot problem from synthetic tasks and verify action dims.
    std::mt19937 rng(7);
    Eigen::VectorXd q, v, tau;
    std::vector<Eigen::VectorXd> xs, us;
    for (int k = 0; k < 3; ++k)
    {
        g1_test::sample_state(model, rng, q, v, tau);
        Eigen::VectorXd x(71);
        x << q, v;
        xs.push_back(x);
        if (k < 2)
            us.push_back(tau);
    }
    auto problem = builder.createEstimationProblem(xs[0], 0.01, 3, xs, us);

    const auto &running = problem->get_runningModels();
    CHECK(running.size() == 3); // arrival + 2 steps
    CHECK(running[0]->get_nu() == 70);
    CHECK(running[1]->get_nu() == 35);
    CHECK(running[2]->get_nu() == 35);
    CHECK(problem->get_terminalModel()->get_nu() == 35);
    for (const auto &m : running)
    {
        CHECK(m->get_state()->get_nx() == 71);
        CHECK(m->get_state()->get_ndx() == 70);
    }

    // Defect dimension: xnext lives on the 71-dim manifold with 70-dim diff.
    auto d0 = running[0]->createData();
    running[0]->calc(d0, xs[0], Eigen::VectorXd::Zero(70));
    CHECK(d0->xnext.size() == 71);
    Eigen::VectorXd dx(70);
    state->diff(d0->xnext, xs[0], dx);
    CHECK(dx.size() == 70);

    std::cout << "test_dimensions OK\n";
    return 0;
}
