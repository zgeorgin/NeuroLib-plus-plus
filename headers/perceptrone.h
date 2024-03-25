#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <ctime>
#include <string>
#include <fstream>

#define SIGMOID 1
#define RELU 2
#define LINEAR 3
#define THRESHOLD 4
#define LEAKY 5
#define TANH 6

double Sigmoid(double value);
double ReLU(double value);
double Linear(double value);
double Threshold(double value);
double Leaky_ReLU(double value);
double Tanh(double value);
void MixDataset(std::vector<std::vector<double>> &features, std::vector<std::vector<double>> &targets);

void ReadCSV(std::string filepath, std::vector<std::vector<double>> &features, std::vector<std::vector<double>> &targets, std::vector<int> targetsColumns, bool oneHotEncode = true);

class Neuron
{
public:
    double value;
    double bias;
    double output;
    Neuron() { value = 0; bias = (double)rand() / (double)RAND_MAX;};
};

class Connection
{
public:
    double weight;
    Neuron *begin;
    Neuron *end;
    Connection(double weight, Neuron *begin, Neuron *end) : weight(weight), begin(begin), end(end){};
};

class Layer
{
public:
    std::vector<Neuron *> neurons;
    std::vector<Connection *> enterConnections;
    Layer *prev = nullptr;
    Layer *next = nullptr;
    int activationFunction;
    Layer(int neuronCount);

    void connect(Layer *another);
};

class Perceptrone
{
public:
    Layer *begin;
    Layer *end;
    bool displacement;
    int activationFunction;
    int layerCount;
    Perceptrone(std::vector<int> neuronCounts, bool displacement, int activationFunction);
    Perceptrone(std::string filepath);

    void fit(std::vector<double> enterNeurons);
    void train(std::vector<double> rightAnswer, double alpha);
    void train(std::vector<std::vector<double>> features, std::vector<std::vector<double>> targets, double alpha);

    void PrintWeights();
    void PrintExit();

    void Save(std::string filepath);

    double Error(std::vector<std::vector<double>> dataset, std::vector<std::vector<double>> rightAnswers);

    double backProp(Neuron *begin, Layer *exit, std::vector<double> &weightDeltas, double &biasDelta, const std::vector<double> &loss);
};
