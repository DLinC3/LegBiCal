// Checkpoint A gate 3: every Pinocchio inertia double of the motion-only model
// is bitwise unchanged around action calls and a complete FDDP solve.
#include "test_common.hpp"

#include <cstring>

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "g1cal/motion_problem.hpp"

namespace
{

std::vector<unsigned char> inertia_bytes(const pinocchio::Model &model)
{
    std::vector<unsigned char> bytes;
    for (const auto &I : model.inertias)
    {
        const Eigen::Matrix<double, 10, 1> pi = I.toDynamicParameters();
        const auto *p = reinterpret_cast<const unsigned char *>(pi.data());
        bytes.insert(bytes.end(), p, p + sizeof(double) * 10);
    }
    return bytes;
}

} // namespace

int main()
{
    auto model = g1_test::load_official_model();
    crocoddyl::ContactIDWeights weights;
    weights.meas_position = 400.0;
    weights.meas_position_z = 200.0;
    weights.meas_orientation = 30.0;
    weights.meas_joint = 200.0;
    weights.meas_linearVel = 0.0;
    weights.meas_angularVel = 150.0;
    weights.meas_jointVel = 40.0;
    weights.dyn_position = 20.0;
    weights.dyn_orientation = 20.0;
    weights.dyn_joint = 20.0;
    weights.kappa = 1000.0;
    weights.mu = 1.0;

    g1cal::MotionFIEProblem builder(model, g1_test::official_contact_frames(),
                                       weights);
    const auto &state_model = *builder.get_state()->get_pinocchio();

    std::mt19937 rng(51);
    std::vector<Eigen::VectorXd> xs_task, us_task;
    Eigen::VectorXd q, v, tau;
    const int knots = 5;
    for (int k = 0; k < knots; ++k)
    {
        g1_test::sample_state(model, rng, q, v, tau, 0.02, 0.05, 2.0);
        Eigen::VectorXd x(71);
        x << q, v;
        xs_task.push_back(x);
        if (k < knots - 1)
            us_task.push_back(tau);
    }

    const auto before = inertia_bytes(state_model);

    auto problem = builder.createEstimationProblem(xs_task[0], 0.01, knots,
                                                   xs_task, us_task);

    // Single action calc/calcDiff round.
    auto m1 = problem->get_runningModels()[1];
    auto d1 = problem->get_runningDatas()[1];
    m1->calc(d1, xs_task[0], us_task[0]);
    m1->calcDiff(d1, xs_task[0], us_task[0]);
    const auto after_action = inertia_bytes(state_model);
    CHECK(after_action.size() == before.size());
    CHECK(std::memcmp(before.data(), after_action.data(), before.size()) == 0);

    // Complete FDDP solve.
    std::vector<Eigen::VectorXd> xs_init;
    xs_init.push_back(xs_task[0]);
    for (int k = 0; k < knots; ++k)
        xs_init.push_back(xs_task[k]);
    std::vector<Eigen::VectorXd> us_init;
    us_init.push_back(Eigen::VectorXd::Zero(70));
    for (auto &u : us_task)
        us_init.push_back(u);

    crocoddyl::SolverFDDP solver(problem);
    solver.solve(xs_init, us_init, 50, false, 0.1);

    const auto after_solve = inertia_bytes(state_model);
    CHECK(std::memcmp(before.data(), after_solve.data(), before.size()) == 0);

    // The source URDF model is also untouched.
    const auto source = inertia_bytes(model);
    CHECK(std::memcmp(before.data(), source.data(), before.size()) == 0);

    std::cout << "test_inertia_invariance OK (bitwise, "
              << before.size() << " bytes)\n";
    return 0;
}
