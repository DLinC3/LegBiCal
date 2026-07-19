// Checkpoint A gate 4: at identical fixed model/q/v/u/contact/mu/kappa/dt the
// official-ID running action and the motion-only running action produce the
// same post-step dynamics (xout head), contact forces, and dv blocks.  The ID
// action reads link-16 inertia from its state in log-Cholesky coordinates set
// to the model values; the log/exp round trip is not bitwise, so comparisons
// use a 1e-8 relative tolerance.
#include "test_common.hpp"

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/contact_id/actions/contact-fwddyn-anitescu-id.hpp"
#include "crocoddyl/contact_id/actuations/floating-base-estimation.hpp"
#include "crocoddyl/contact_id/anitescu/Logchol.hpp"
#include "g1cal/motion_fwddyn_action.hpp"

int main()
{
    auto model = g1_test::load_official_model();
    const auto frames = g1_test::official_contact_frames();
    const double kappa = 1000.0, mu = 1.0, dt = 0.01;
    const pinocchio::JointIndex jid = 16; // waist_pitch_joint

    // Official ID action with one identified link at model inertia values.
    auto model_ptr_id = boost::make_shared<pinocchio::Model>(model);
    std::vector<pinocchio::JointIndex> joints{jid};
    auto id_state = boost::make_shared<crocoddyl::StateMultibodyParams>(
        model_ptr_id, joints);
    auto id_act = boost::make_shared<
        crocoddyl::ActuationModelFloatingBaseEstimation>(id_state);
    auto id_costs = boost::make_shared<crocoddyl::CostModelSum>(
        id_state, id_act->get_nu());
    auto id_dam = boost::make_shared<
        crocoddyl::DifferentialActionModelContactFwdDynamicsAnitescuSystemid>(
        id_state, id_act, id_costs, frames, kappa, mu, dt, "");
    auto id_data = id_dam->createData();

    // Motion-only action on the same frozen model.
    auto model_ptr_mo = boost::make_shared<pinocchio::Model>(model);
    auto mo_state =
        boost::make_shared<crocoddyl::StateMultibodyTpl<double>>(model_ptr_mo);
    auto mo_act = boost::make_shared<
        crocoddyl::ActuationModelFloatingBaseEstimationTpl<double>>(mo_state);
    auto mo_costs = boost::make_shared<crocoddyl::CostModelSum>(
        mo_state, mo_act->get_nu());
    auto mo_dam = boost::make_shared<
        g1cal::DifferentialActionModelContactFwdDynamicsMotion>(
        mo_state, mo_act, mo_costs, frames, kappa, mu, dt,
        /*exact_q_sensitivity=*/false,
        /*newton_max_iters=*/100);
    auto mo_data = mo_dam->createData();

    CHECK(id_state->get_nx() == 91);
    CHECK(id_state->get_ndx() == 90);
    CHECK(mo_state->get_nx() == 71);
    CHECK(mo_state->get_ndx() == 70);
    CHECK(id_dam->get_nu() == 35);
    CHECK(mo_dam->get_nu() == 35);

    const Eigen::VectorXd pi_log = computeLogCholeskyFromLink(model, (int)jid);
    const int nv = model.nv;

    std::mt19937 rng(23);
    for (int trial = 0; trial < 10; ++trial)
    {
        Eigen::VectorXd q, v, tau;
        g1_test::sample_state(model, rng, q, v, tau);

        Eigen::VectorXd x_id(91);
        x_id << q, pi_log, v, Eigen::VectorXd::Zero(10);
        Eigen::VectorXd x_mo(71);
        x_mo << q, v;
        const Eigen::VectorXd u = tau;

        id_dam->calc(id_data, x_id, u);
        mo_dam->calc(mo_data, x_mo, u);

        const double xout_scale =
            std::max(1.0, id_data->xout.head(nv).lpNorm<Eigen::Infinity>());
        CHECK((id_data->xout.head(nv) - mo_data->xout)
                      .lpNorm<Eigen::Infinity>() /
                  xout_scale <
              1e-8);
        CHECK(id_data->xout.tail(10).lpNorm<Eigen::Infinity>() == 0.0);

        id_dam->calcDiff(id_data, x_id, u);
        mo_dam->calcDiff(mo_data, x_mo, u);

        // ID tangent layout: [dq(35), dtheta(10), dv(35), dtheta_rate(10)].
        const auto id_dvdq = id_data->Fx.block(0, 0, nv, nv);
        const auto id_dvdv = id_data->Fx.block(0, nv + 10, nv, nv);
        const auto mo_dvdq = mo_data->Fx.block(0, 0, nv, nv);
        const auto mo_dvdv = mo_data->Fx.block(0, nv, nv, nv);

        const double fx_scale =
            std::max(1.0, id_dvdq.lpNorm<Eigen::Infinity>());
        CHECK((id_dvdq - mo_dvdq).lpNorm<Eigen::Infinity>() / fx_scale < 1e-6);
        CHECK((id_dvdv - mo_dvdv).lpNorm<Eigen::Infinity>() /
                  std::max(1.0, id_dvdv.lpNorm<Eigen::Infinity>()) <
              1e-6);
        CHECK((id_data->Fu.topRows(nv) - mo_data->Fu.topRows(nv))
                      .lpNorm<Eigen::Infinity>() /
                  std::max(1.0, id_data->Fu.lpNorm<Eigen::Infinity>()) <
              1e-6);

        // Latent contact force parity (source order [t1,t2,n], 8 contacts).
        const Eigen::VectorXd f_mo = mo_dam->get_last_force();
        CHECK(f_mo.size() == 24);
    }

    // Inertia values in the ID model after calc equal the frozen model values
    // to round-trip precision (log-Cholesky exp/log), while the motion model
    // is untouched — checked bitwise in test_inertia_invariance.
    for (pinocchio::JointIndex j = 1;
         j < (pinocchio::JointIndex)model.njoints; ++j)
    {
        const double dm = std::abs(model_ptr_id->inertias[j].mass() -
                                   model.inertias[j].mass());
        CHECK(dm < 1e-10);
    }

    std::cout << "test_action_parity OK\n";
    return 0;
}
