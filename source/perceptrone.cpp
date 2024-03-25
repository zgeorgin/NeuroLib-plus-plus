#include "../headers/perceptrone.h"

double Activation(double value, int activationFunction)
{
    switch (activationFunction)
    {
    case SIGMOID:
        return Sigmoid(value);
    case RELU:
        return ReLU(value);
    case LINEAR:
        return Linear(value);
    case THRESHOLD:
        return Threshold(value);
    case LEAKY:
        return Leaky_ReLU(value);
    case TANH:
        return Tanh(value);
    default:
        return value;
    }
}

double Derivative(double value, int activationFunction)
{
    switch (activationFunction)
    {
    case SIGMOID:
        return Sigmoid(value) * (1 - Sigmoid(value));
    case RELU:
        if (value > 0)
            return 1;
        return 0;
    case LINEAR:
        return 1;
    case THRESHOLD:
        return 1;
    case LEAKY:
        if (value > 0)
            return 1;
        return 0.01;
    case TANH:
        return 1 - Tanh(value) * Tanh(value);
    default:
        return 1;
    }
}

double Sigmoid(double value)
{
    return 1 / (1 + exp(-value));
}

double ReLU(double value)
{
    if (value < 0)
        return 0;
    return value;
}

double Leaky_ReLU(double value)
{
    if (value < 0)
        return value * 0.01;
    return value;
}

double Linear(double value)
{
    return value;
}

double Threshold(double value)
{
    if (value <= 0.5)
        return 0;
    return 1;
}

double Tanh(double value) { return tanh(value); }

Layer::Layer(int neuronCount)
{
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
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<> distr(-sqrt(3), sqrt(3));
    for (Neuron *n1 : another->neurons)
    {
        for (Neuron *n2 : neurons)
        {

            double weight = (double)rand() / ((double)RAND_MAX);//distr(gen); 
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
        current->neurons[i]->output = Activation(enterNeurons[i], activationFunction);

    if (displacement)
        current->neurons[current->neurons.size() - 1]->output = Activation(1, activationFunction);

    current = current->next;
    while (current != nullptr)
    {
        for (Neuron *n : current->neurons)
            n->value = 0;

        for (Connection *c : current->enterConnections)
            c->end->value += c->begin->output * c->weight;

        for (Neuron *n : current->neurons)
            n->output = Activation(n->value + n->bias, activationFunction);

        current = current->next;
    }
}

double Perceptrone::backProp(Neuron *begin, Layer *exit, std::vector<double> &weightDeltas, double &biasDelta, const std::vector<double> &loss)
{
    double beginLoss = 0;
    for (int i = 0; i < exit->enterConnections.size(); i++)
    {
        Connection *c = exit->enterConnections[i];
        if (c->begin == begin)
        {
            for (int j = 0; j < exit->neurons.size(); j++)
            {
                weightDeltas[i] += c->begin->output * Derivative(c->end->output, activationFunction) * loss[j] * (c->end == exit->neurons[j]);
                biasDelta += Derivative(c->end->output, activationFunction) * loss[j] * (c->end == exit->neurons[j]);
                beginLoss += c->weight * Derivative(c->end->output, activationFunction) * loss[j] * (c->end == exit->neurons[j]);
            }
        }
    }
    return beginLoss;
}

void Perceptrone::train(std::vector<double> rightAnswer, double alpha)
{
    std::vector<double> loss(end->neurons.size());
    for (int i = 0; i < loss.size(); i++)
        loss[i] = 2 * (end->neurons[i]->output - rightAnswer[i]);

    Layer *current = end;
    while (current != begin)
    {
        std::vector<double> weightDeltas(current->enterConnections.size(), 0);
        std::vector<double> newLoss(current->prev->neurons.size());

        for (int i = 0; i < current->prev->neurons.size(); i++)
        {
            double biasDelta = 0;
            newLoss[i] = backProp(current->prev->neurons[i], current, weightDeltas, biasDelta, loss);
            current->prev->neurons[i]->bias -= biasDelta * alpha;
        }

        for (int i = 0; i < current->enterConnections.size(); i++)
            current->enterConnections[i]->weight -= weightDeltas[i] * alpha;

        loss = newLoss;
        current = current->prev;
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
        std::cout << n->output << ' ';

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

void Perceptrone::train(std::vector<std::vector<double>> features, std::vector<std::vector<double>> rightAnswers, double alpha)
{
    for (int i = 0; i < features.size(); i++)
    {
        fit(features[i]);
        train(rightAnswers[i], alpha);
    }
}

double Perceptrone::Error(std::vector<std::vector<double>> features, std::vector<std::vector<double>> rightAnswers)
{
    double error = 0;
    for (int i = 0; i < features.size(); i++)
    {
        fit(features[i]);
        for (int j = 0; j < rightAnswers[i].size(); j++)
            error += (end->neurons[j]->output - rightAnswers[i][j]) * (end->neurons[j]->output - rightAnswers[i][j]);
    }
    return error/features.size();
}

void MixDataset(std::vector<std::vector<double>> &features, std::vector<std::vector<double>> &targets)
{
    for (int i = 0; i < targets.size(); i++)
    {
        int pivot = rand() % features.size();
        std::swap(features[i], features[pivot]);
        std::swap(targets[i], targets[pivot]);
    }
}

void ReadCSV(std::string filepath, std::vector<std::vector<double>> &features, std::vector<std::vector<double>> &targets, std::vector<int> targetsColumns, bool oneHotEncode)
{
    /*std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> rowFeatures;
        double target;
        int columnNumber = 0;

        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (columnNumber == targetColumn) {
                target = std::stod(cell);
            } else if (std::find(featureColumns.begin(), featureColumns.end(), columnNumber) != featureColumns.end()) {
                rowFeatures.push_back(std::stod(cell));
            }
            columnNumber++;
        }

        features.push_back(rowFeatures);
        targets.push_back(target);
    }

    file.close();*/
}