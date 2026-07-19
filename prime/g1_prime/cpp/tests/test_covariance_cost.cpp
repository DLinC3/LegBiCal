// Stage 3: strict discrete costs are exactly 0.5*r'precision*r for arrival,
// running measurement/process, and terminal measurement despite Euler scaling.
#include "test_common.hpp"

#include "g1cal/motion_problem.hpp"

int main()
{
    auto model = g1_test::load_official_model();
    crocoddyl::ContactIDWeights legacy;
    legacy.kappa = 1000.;
    legacy.mu = 1.;

    g1cal::DiagonalCovariancePrecision covariance;
    covariance.enabled = true;
    covariance.config_hash = "unit-test";
    covariance.p0 = Eigen::VectorXd::Constant(70, 2.0);
    covariance.q = Eigen::VectorXd::Constant(35, 5.0);
    covariance.r = Eigen::VectorXd::Constant(70, 3.0);

    g1cal::MotionFIEProblem builder(model, g1_test::official_contact_frames(),
                                       legacy, covariance);
    auto state = builder.get_state();
    std::mt19937 rng(93);
    Eigen::VectorXd q, v, tau;
    g1_test::sample_state(model, rng, q, v, tau, 0.01, 0.01, 0.1);
    Eigen::VectorXd task(71);
    task << q, v;
    std::vector<Eigen::VectorXd> tasks{task, task};
    std::vector<Eigen::VectorXd> controls{tau};
    const double dt = 0.01;
    auto problem = builder.createEstimationProblem(task, dt, 2, tasks, controls);

    Eigen::VectorXd dx = Eigen::VectorXd::LinSpaced(70, -1e-3, 1e-3);
    Eigen::VectorXd xpert(71);
    state->integrate(task, dx, xpert);
    Eigen::VectorXd actual_dx(70);
    state->diff(task, xpert, actual_dx);

    // Arrival cost on its 70-dim tangent control.
    auto arrival = problem->get_runningModels()[0];
    auto arrival_data = arrival->createData();
    arrival->calc(arrival_data, task, dx);
    const double expected_arrival =
        0.5 * (covariance.p0.array() * dx.array().square()).sum();
    CHECK_NEAR(arrival_data->cost, expected_arrival, 1e-12);

    // One controlled running model: state and generalized-force residuals.
    auto running = problem->get_runningModels()[1];
    auto running_data = running->createData();
    Eigen::VectorXd du = Eigen::VectorXd::LinSpaced(35, -2e-3, 2e-3);
    running->calc(running_data, xpert, tau + du);
    const double expected_state =
        0.5 * (covariance.r.array() * actual_dx.array().square()).sum();
    const double expected_control =
        0.5 * (covariance.q.array() * du.array().square()).sum();
    CHECK_NEAR(running_data->cost, expected_state + expected_control, 1e-10);

    // Terminal calc(x) is not Euler-dt scaled, so its builder coefficient is 1.
    auto terminal = problem->get_terminalModel();
    auto terminal_data = terminal->createData();
    terminal->calc(terminal_data, xpert);
    CHECK_NEAR(terminal_data->cost, expected_state, 1e-12);

    std::cout << "test_covariance_cost OK\n";
    return 0;
}
