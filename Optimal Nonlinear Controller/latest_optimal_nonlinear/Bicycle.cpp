#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <mlpack.hpp>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "Header.h"



// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;



void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {

    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}



void mouse_button(GLFWwindow* window, int button, int act, int mods) {

    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}


void mouse_move(GLFWwindow* window, double xpos, double ypos) {

    if (!button_left && !button_middle && !button_right) {
        return;
    }

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;


    int width, height;
    glfwGetWindowSize(window, &width, &height);


    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) == GLFW_PRESS;
    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;


    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}



void scroll(GLFWwindow* window, double xoffset, double yoffset) {

    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}


using namespace arma;
using namespace ens;
using namespace mlpack;
using namespace std;



extern mjtNum force_scale[2];

extern FFN<EmptyLoss, GaussianInitialization>* policy = nullptr;
mat S(3, 1); // eta, eta_dot, eta_int
mat A(1, 1);

mjtNum theta = 0.0;
mjtNum theta_dot = 0.0;
mjtNum theta_int = 0.0;
float x = 0.0;
float y = 0.0;
float eta = 0.0;
float eta_dot = 0.0;
float eta_int = 0.0;
float eta_old = 0.0;

float v1 = 0.0;
float v2 = 0.0;

mat N(3, 1, fill::zeros);
mat Z(3, 1, fill::zeros);
mat W(3, 1, fill::zeros);


float leakyReLU(float v) {
    static const float alpha = 0.5;
    return ((v > 0.0) ? v : alpha * v);
}


void test_controller(const mjModel* m, mjData* d) {

    S(0, 0) = eta;
    S(1, 0) = eta_dot;
    S(2, 0) = eta_int;

    policy->Predict(S, A);

    d->ctrl[0] = 40.0;
    d->ctrl[1] = force_scale[1] * A(0, 0);
    d->ctrl[2] = 300.0 * (51.4397 * theta + 50.9348 * theta_int + 11.5886 * theta_dot);
}



int main() {

    int option = 0;
    cin >> option;

    if (option == 0) {
        show();
    }
    else if (option == 1) {
        load();
    }
    
    train();

    cout << "Weights:" << endl;

    vector<Layer<mat>*> layers = policy->Network();
    policy->Parameters().print();
    layers[0]->Parameters().print();

    ofstream F("test.txt", ios_base::trunc);


    char error[1000];
    m = mj_loadXML("Bicycle.xml", 0, error, 1000);

    // make data
    d = mj_makeData(m);


    glfwInit();


    GLFWwindow* window = glfwCreateWindow(1200, 900, "Bicycle", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);


    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);


    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);


    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);


    mjcb_control = test_controller;
    

    Z(2, 0) = 1.0;

    system("color A");


    while (!glfwWindowShouldClose(window)) {

        mjtNum simstart = d->time;


        while (d->time - simstart < 1.0 / 60.0) {

            N(0, 0) = d->sensordata[3];
            N(1, 0) = d->sensordata[4];
            N(2, 0) = d->sensordata[5];

            W(0, 0) = d->sensordata[6];
            W(1, 0) = d->sensordata[7];
            W(2, 0) = d->sensordata[8];

            theta = acos(dot(Z, N)) - M_PI / 2;
            theta_dot = dot(W, cross(Z, N));
            theta_int += theta * m->opt.timestep;
            
            x = d->sensordata[0];
            y = d->sensordata[1];

            eta_old = eta;
            eta = dR(x, y) / (dR(x, y) + dL(x, y)) - 0.5;
            eta_int += eta * m->opt.timestep;
            eta_dot = (eta - eta_old) / m->opt.timestep;

            mj_step(m, d);

        }

        F << d->time << ',' << x << ',' << y << ',' << eta << ',' << d->ctrl[1] << endl;

        v1 = -7.2402 * eta + 1.5795 * eta_dot - 24.6205 * eta_int;
        v2 = 13.2579 * eta + 3.0876 * eta_dot + 21.9366 * eta_int;

        //system("cls");

        if ((v1 < 0) && (v2 < 0)) {
            cout << "Subspace 1";
        }

        if ((v1 > 0) && (v2 < 0)) {
            cout << "Subspace 2";
        }

        if ((v1 < 0) && (v2 > 0)) {
            cout << "Subspace 3";
        }

        if ((v1 > 0) && (v2 > 0)) {
            cout << "Subspace 4";
        }


        if ((d->sensordata[9] < -0.1) || (d->sensordata[10] < -0.1)) {
            F.clear();
            F.close();
            break;
        }


        mjrRect viewport = { 0, 0, 0, 0 };
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);


        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);


        glfwSwapBuffers(window);


        glfwPollEvents();
    }


    mjv_freeScene(&scn);
    mjr_freeContext(&con);


    mj_deleteData(d);
    mj_deleteModel(m);


#if defined(__APPLE__)  defined(_WIN32)
    glfwTerminate();
#endif

    return 0;
}