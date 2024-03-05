#include "perceptrone.h"

void Neuron::Activation(int activationFunction)
{
    switch (activationFunction)
    {
    case SIGMOID:
        value = Sigmoid(value);
        break;
    case RELU:
        value = ReLU(value);
        break;
    case LINEAR:
        value = Linear(value);
        break;
    case THRESHOLD:
        value = Threshold(value);
        break;
    case LEAKY:
        value = Leaky_ReLU(value);
        break;
    }
}

double Neuron::Sigmoid(double value)
{
    return 1 / (1 + exp(-value));
}

double Neuron::ReLU(double value)
{
    if (value < 0)
        return 0;
    return value;
}

double Neuron::Leaky_ReLU(double value)
{
    if (value < 0)
        return value*0.01;
    return value;
}

double Neuron::Linear(double value)
{
    return value;
}

double Neuron::Threshold(double value)
{
    if (value <= 0)
        return 0;
    return 1;
}

Layer::Layer(int neuronCount)
{
    srand(time(0));
    for (int i = 0; i < neuronCount; i++)
    {
        Neuron *n = new Neuron();
        neurons.push_back(n);
    }
}

void Layer::connect(Layer *another)
{
    another->prev = this;
    next = another;
    for (Neuron *n1 : another->neurons)
    {
        for (Neuron *n2 : neurons)
        {
            double weight = (double)rand() / (double)RAND_MAX;
            Connection *c = new Connection(weight, n2, n1);
            another->enterConnections.push_back(c);
        }
    }
}

Perceptrone::Perceptrone(std::vector<int> neuronCounts, bool displacement = false, int activationFunction = LINEAR)
{
    this->layerCount = neuronCounts.size();
    this->displacement = displacement;
    begin = new Layer(neuronCounts[0] + displacement);
    Layer *current = begin;
    this->activationFunction = activationFunction;
    for (int i = 1; i < layerCount; i++)
    {
        Layer *next = new Layer(neuronCounts[i]);
        current->connect(next);
        if (i == layerCount - 1)
            end = next;
        current = next;
    }
}

void Perceptrone::fit(std::vector<double> enterNeurons)
{
    Layer *current = begin;
    for (int i = 0; i < enterNeurons.size(); i++)
    {
        current->neurons[i]->value = enterNeurons[i];
        current->neurons[i]->Activation(activationFunction);
    }

    if (displacement)
        current->neurons[current->neurons.size() - 1]->value = 1;

    current = current->next;
    while (current != nullptr)
    {

        for (Neuron *n : current->neurons)
            n->value = 0;

        for (Connection *c : current->enterConnections)
            c->end->value += c->begin->value * c->weight;

        for (Neuron *n : current->neurons)
            n->Activation(activationFunction);

        current = current->next;
    }
}

void Perceptrone::train(std::vector<double> rightAnswer, double alpha)
{
    Layer *current;
    int layerNumber = 0;
    std::vector<std::vector<double>> weightDeltas(layerCount - 1);
    std::vector<double> delta(rightAnswer.size());

    for (int i = 0; i < rightAnswer.size(); i++)
    {
        double pred = end->neurons[i]->value;
        delta[i] = pred - rightAnswer[i];
    }
    current = end;

    double nextDelta = 0;
    for (int i = 0; i < rightAnswer.size(); i++)
    {
        for (Connection *c : current->enterConnections)
            nextDelta += delta[i] * c->weight * (c->end == end->neurons[i]);
    }

    std::vector<double> layerDeltas = {nextDelta};
    current = current->prev;
    while (current->prev != nullptr)
    {
        double delta = 0;
        for (Connection *c : current->enterConnections)
            delta += layerDeltas[layerDeltas.size() - 1] * c->weight;

        layerDeltas.push_back(delta);
        current = current->prev;
    }

    current = current->next;
    layerNumber = 0;
    while (current->next != nullptr)
    {
        for (int i = 0; i < current->enterConnections.size(); i++)
            weightDeltas[layerNumber].push_back(current->enterConnections[i]->begin->value * layerDeltas[layerDeltas.size() - 1 - layerNumber]);

        layerNumber++;
        current = current->next;
    }

    for (int i = 0; i < current->enterConnections.size(); i++)
    {
        for (int j = 0; j < current->neurons.size(); j++)
        {
            if (j == 0)
            {
                weightDeltas[layerNumber].push_back(current->enterConnections[i]->begin->value * delta[j] * (current->enterConnections[i]->end == current->neurons[j]));
            }
            else
                weightDeltas[layerNumber][i] += current->enterConnections[i]->begin->value * delta[j] * (current->enterConnections[i]->end == current->neurons[j]);
        }
    }

    current = begin->next;
    layerNumber = 0;
    while (current != nullptr)
    {
        for (int i = 0; i < current->enterConnections.size(); i++)
            current->enterConnections[i]->weight -= weightDeltas[layerNumber][i] * alpha;

        layerNumber++;
        current = current->next;
    }
}

void Perceptrone::PrintWeights()
{
    Layer *current = begin->next;
    while (current != nullptr)
    {
        for (Connection *c : current->enterConnections)
        {
            std::cout << c->weight << ' ';
        }
        std::cout << '\n';
        current = current->next;
    }
}

void Perceptrone::PrintExit()
{
    for (Neuron *n : end->neurons)
        std::cout << n->value << ' ';

    std::cout << '\n';
}

void Perceptrone::Save(std::string filepath)
{
    std::ofstream out;
    out.open(filepath);

    if (out.is_open())
    {
        Layer *current = begin;
        out << layerCount << '\n';
        while (current != nullptr)
        {
            out << current->neurons.size() << '\n';
            current = current->next;
        }

        out << displacement << '\n';
        out << activationFunction << '\n';

        current = begin->next;
        while (current != nullptr)
        {
            for (Connection *c : current->enterConnections)
            {
                out << c->weight << '\n';
            }
            current = current->next;
        }
    }

    out.close();
}

Perceptrone::Perceptrone(std::string filepath)
{
    std::string line;
    std::vector<int> neuronCounts;
    std::ifstream in(filepath);
    if (in.is_open())
    {
        std::getline(in, line);
        layerCount = stoi(line);

        for (int i = 0; i < layerCount; i++)
        {
            std::getline(in, line);
            neuronCounts.push_back(stoi(line));
        }

        std::getline(in, line);
        displacement = (bool)stoi(line);

        std::getline(in, line);
        activationFunction = stoi(line);

        begin = new Layer(neuronCounts[0]);

        Layer *current = begin;
        for (int i = 1; i < layerCount; i++)
        {
            Layer *next = new Layer(neuronCounts[i]);
            current->connect(next);
            if (i == layerCount - 1)
                end = next;
            current = next;
            for (Connection *c : current->enterConnections)
            {
                std::getline(in, line);
                c->weight = stod(line);
            }
        }
    }
    in.close();
}

void Perceptrone::train(std::vector<std::vector<double>> dataset, std::vector<std::vector<double>> rightAnswers, double alpha)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        fit(dataset[i]);
        train(rightAnswers[i], alpha);
    }
}

double Perceptrone::Error(std::vector<std::vector<double>> dataset, std::vector<std::vector<double>> rightAnswers)
{
    double error = 0;
    for (int i = 0; i < dataset.size(); i++)
    {
        fit(dataset[i]);
        for (int j = 0; j < rightAnswers[i].size(); j++)
            error += (end->neurons[j]->value - rightAnswers[i][j]) * (end->neurons[j]->value - rightAnswers[i][j]);
    }
    return error;
}

void MixDataset (std::vector<std::vector<double>>& dataset, std::vector<std::vector<double>>& features)
{
    for (int i = 0; i < dataset.size(); i++)
    {
        int pivot = rand() % dataset.size();
        std::swap(dataset[i], dataset[pivot]);
        std::swap(features[i], features[pivot]);
    }
}

void ReadCSV(std::string filename, std::vector<std::vector<double>>& dataset, std::vector<std::vector<double>> features, std::vector<int> featuresColumns)
{
    
}