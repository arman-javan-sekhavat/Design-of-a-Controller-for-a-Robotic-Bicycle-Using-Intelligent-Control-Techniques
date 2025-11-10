#include "Header.h"
#include "environment.h"

#include <fstream>

using namespace arma;
using namespace mlpack;
using namespace std;
using namespace ens;


extern FFN<EmptyLoss, GaussianInitialization>* policy;



void train(void) {

    mjcb_control = controller;

    // Set up the replay method.
    RandomReplay<Bicycle> replayMethod(30, 700);

    // Set up the training configuration.
    TrainingConfig config;
    config.StepSize() = 0.01;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;
    config.Discount() = 0.90;

    // Set up Actor network.
    static FFN<EmptyLoss, GaussianInitialization>  policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
    policy = &policyNetwork;
    policyNetwork.Add(new LinearNoBias(2));
    policyNetwork.Add(new LeakyReLU(0.5));
    policyNetwork.Add(new LinearNoBias(1));
    policyNetwork.Add(new HardTanH());


    // Set up Critic network.
    FFN<EmptyLoss, GaussianInitialization>  qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
    qNetwork.Add(new Linear(12));
    qNetwork.Add(new ReLU());
    qNetwork.Add(new Linear(10));
    qNetwork.Add(new ReLU());
    qNetwork.Add(new Linear(1));

    // Set up the GaussianNoise parameters.
    int size = 1;
    double mu = 0.0;
    double sigma = 1.0;

    // Create an instance of the GaussianNoise class.
    GaussianNoise gaussianNoise(size, mu, sigma);

    // Set up Deep Deterministic Policy Gradient agent.
    DDPG<Bicycle, decltype(qNetwork), decltype(policyNetwork),
        GaussianNoise, AdamUpdate>
        agent(config, qNetwork, policyNetwork, gaussianNoise, replayMethod);


    std::vector<double> lastEpisodes;
    double R;
    double sum = 0.0;
    double maxAvg = 0.0;

    ofstream F("weights.txt", ios_base::trunc);
    mat W;

    for (episode = 1; episode < 30; episode++) {

        if (lastEpisodes.size() == 15) {
            lastEpisodes.erase(lastEpisodes.begin());
        }
        
        R = agent.Episode();
        W = policy->Parameters();

        for (int i = 0; i < 8; i++) {
            F << W[i] << ',';
        }

        F << endl;


        lastEpisodes.push_back(R);



        sum = 0.0;

        for (int j = 0; j < 15; j++) {
            sum += lastEpisodes[j];
        }

        sum /= 15;

        std::cout << "Episode: " << episode << ", Return: " << R << endl;

        if (episode > 25) {
            maxAvg = (sum > maxAvg) ? sum : maxAvg;
        }

        /*if ((episode > 25) && (lastEpisodes.size() == 15) && (sum > 598.406)) {
            break;
        }*/


    }

    F.clear();
    F.close();

    std::cout << "Maximum Average Return: " << maxAvg;

    agent.Deterministic() = true;
    std::cout << endl << "Test" << endl;
    std::cout << agent.Episode() << endl;


}