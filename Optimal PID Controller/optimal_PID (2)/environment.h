#pragma once

#include <mlpack/prereqs.hpp>
#include <mlpack.hpp>
#include <mujoco/mujoco.h>
#include <fstream>
#include <string>


extern mjtNum force_scale[2] = {200.0, 500.0};
mjtNum force[2] = {0.0};


float theta = 0.0;
float theta_dot = 0.0;
float theta_int = 0.0;
int episode = 0;

void controller(const mjModel* m, mjData* d) {

    d->ctrl[0] = 40.0;
    d->ctrl[1] = force_scale[1]*force[1];
    d->ctrl[2] = 300.0 * (51.4397 * theta + 50.9348*theta_int + 11.5886 * theta_dot);
}


using namespace arma;


namespace mlpack {

    class Bicycle
    {
    public:
        /**
         * Implementation of the state of Bicycle. Each state is a tuple vector
         */
        class State
        {
        public:
            /**
             * Construct a state instance.
             */
            State() : data(dimension)
            { /* Nothing to do here. */
            }

            /**
             * Construct a state instance from given data.
             *
             * @param data Data for the position, velocity, angle and angular velocity.
             */
            State(const arma::colvec& data) : data(data)
            { /* Nothing to do here */
            }

            //! Modify the internal representation of the state.
            arma::colvec& Data() { return data; }


            //! Get the angle.
            double eta() const { return data[0]; }
            //! Modify the angle.
            double& eta() { return data[0]; }

            //! Get the angular velocity.
            double eta_dot() const { return data[1]; }
            //! Modify the angular velocity.
            double& eta_dot() { return data[1]; }


            double eta_int() const { return data[2]; }
            double& eta_int() { return data[2]; }

            //! Encode the state to a column vector.
            const arma::colvec& Encode() const { return data; }

            //! Dimension of the encoded state.
            static constexpr size_t dimension = 3;

        private:
            //! Locally-stored (eta, eta_dot).
            arma::colvec data;
        };

        /**
         * Implementation of action of Bicycle.
         */
        class Action
        {
        public:
            // To store the action.

            Action() : action(1)
            { /* Nothing to do here */
            }

            std::vector<double> action;

            // Number of degrees of freedom.
            static const size_t size = 1;
        };

        /**
         * Construct a Bicycle instance using the given constants.
         *
         * @param maxSteps The number of steps after which the episode
         *    terminates. If the value is 0, there is no limit.
         * @param tau The time interval.
         * @param thetaThresholdRadians The maximum angle.
         * @param doneReward Reward recieved by agent on success.
         */
        Bicycle(const size_t maxSteps = 0,
            double tau = 0.001,
            const double thetaThresholdRadians = 15 * 2 * 3.1416 / 360,
            const double doneReward = 1.0
        ) :
            maxSteps(maxSteps),
            tau(tau),
            thetaThresholdRadians(thetaThresholdRadians),
            doneReward(doneReward),
            stepsPerformed(0)
        {
            char error[1000];
            m = mj_loadXML("Bicycle.xml", 0, error, 1000);
            d = mj_makeData(m);
            tau = m->opt.timestep;

            N = mat(3, 1, fill::zeros);
            Z = mat(3, 1, fill::zeros);
            W = mat(3, 1, fill::zeros);

            Z(2, 0) = 1.0;
        }

        /**
         * Dynamics of Bicycle instance. Get reward and next state based on current
         * state and current action.
         *
         * @param state The current state.
         * @param action The current action.
         * @param nextState The next state.
         * @return reward.
         */
        double Sample(const State& state,
            const Action& action,
            State& nextState)
        {
            // Update the number of steps performed.
            stepsPerformed++;

            force[0] = action.action[0];
            force[1] = action.action[0];

            // Update states.
            mj_step(m, d);

            x = d->sensordata[0];
            y = d->sensordata[1];

            N(0, 0) = d->sensordata[3];
            N(1, 0) = d->sensordata[4];
            N(2, 0) = d->sensordata[5];

            W(0, 0) = d->sensordata[6];
            W(1, 0) = d->sensordata[7];
            W(2, 0) = d->sensordata[8];


            eta_old = eta;
            eta = dR(x, y) / (dR(x, y) + dL(x, y)) - 0.5;
            eta_int += eta * tau;
            eta_dot = (eta - eta_old) / tau;
            theta = acos(dot(Z, N)) - 0.5 * M_PI;
            theta_dot = dot(W, cross(Z, N));
            theta_int += theta * tau;
            theta_d = theta * (180 / M_PI);

            nextState.eta() = eta;
            nextState.eta_dot() = eta_dot;
            nextState.eta_int() = eta_int;


            // Check if the episode has terminated.
            bool done = IsTerminal(nextState);


            return 3.0 / cosh(2*sqrt(eta*eta + eta_int*eta_int));
        }

        /**
         * Dynamics of Cart Pole. Get reward based on current state and current
         * action.
         *
         * @param state The current state.
         * @param action The current action.
         * @return reward, it's always 1.0.
         */
        double Sample(const State& state, const Action& action)
        {
            State nextState;
            return Sample(state, action, nextState);
        }

        /**
         * Initial state representation is randomly generated within [-0.05, 0.05].
         *
         * @return Initial state for each episode.
         */
        State InitialSample()
        {
            stepsPerformed = 0;

            arma::colvec initialPose = arma::colvec(3, arma::fill::zeros);
            initialPose(0) = 0.0;

            return State(initialPose);

        }

        /**
         * This function checks if the Bicycle has reached the terminal state.
         *
         * @param state The desired state.
         * @return true if state is a terminal state, otherwise false.
         */
        bool IsTerminal(const State& state)
        {
            if (maxSteps != 0 && stepsPerformed >= maxSteps)
            {
                Log::Info << "Episode terminated due to the maximum number of steps"
                    "being taken.";

                mj_resetData(m, d);
                return true;
            }
            else if ((std::fabs(theta) > thetaThresholdRadians) ||
                     (d->sensordata[9] < -0.1) ||
                     (d->sensordata[10] < -0.1) ||
                     (y < -6.0) )
            {
                Log::Info << "Episode terminated due to agent failing.";

                eta_int = 0.0;
                mj_resetData(m, d);
                theta_int = 0.0;
                return true;
            }
            return false;
        }

        //! Get the number of steps performed.
        size_t StepsPerformed() const { return stepsPerformed; }

        //! Get the maximum number of steps allowed.
        size_t MaxSteps() const { return maxSteps; }
        //! Set the maximum number of steps allowed.
        size_t& MaxSteps() { return maxSteps; }

    private:

        mat N;
        mat Z;
        mat W;

        float x = 0.0;
        float y = 0.0;

        float eta = 0.0;
        float eta_dot = 0.0;
        float eta_int = 0.0;
        float eta_old = 0.0;

        float theta_d = 0.0;

        mjModel* m = NULL;
        mjData*  d = NULL;

        //! Locally-stored maximum number of steps.
        size_t maxSteps;

        //! Locally-stored time interval.
        double tau;

        //! Locally-stored maximum angle.
        double thetaThresholdRadians;

        //! Locally-stored done reward.
        double doneReward;

        //! Locally-stored number of steps performed.
        size_t stepsPerformed;

    };

} // namespace mlpack