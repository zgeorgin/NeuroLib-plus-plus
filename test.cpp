#include "perceptrone.h"

void test_AND()
{
    std::vector<int> neuronCounts = {2, 1};
    Perceptrone *p = new Perceptrone(neuronCounts, true, THRESHOLD);

    std::vector<std::vector<double>> dataset;
    for (int i = 0; i <= 1; i++)
    {
        for (int j = 0; j <= 1; j++)
        {
            std::vector<double> data = {(double)i, (double)j};
            dataset.push_back(data);
        }
    }
    std::vector<double> ans1 = {0};
    std::vector<double> ans2 = {0};
    std::vector<double> ans3 = {0};
    std::vector<double> ans4 = {1};

    std::vector<std::vector<double>> rightAnswers = {ans1, ans2, ans3, ans4};
    for (int i = 0; i < 100; i++)
    {
        p->train(dataset, rightAnswers, 0.01);
    }

    for (std::vector<double> data : dataset)
    {
        p->fit(data);
        p->PrintExit();
    }

    p->PrintWeights();

    delete p;
}

void test_OR()
{
    std::vector<int> neuronCounts = {2, 1};
    Perceptrone *p = new Perceptrone(neuronCounts, true, THRESHOLD);

    std::vector<std::vector<double>> dataset;
    for (int i = 0; i <= 1; i++)
    {
        for (int j = 0; j <= 1; j++)
        {
            std::vector<double> data = {(double)i, (double)j};
            dataset.push_back(data);
        }
    }
    std::vector<double> ans1 = {0};
    std::vector<double> ans2 = {1};
    std::vector<double> ans3 = {1};
    std::vector<double> ans4 = {1};

    std::vector<std::vector<double>> rightAnswers = {ans1, ans2, ans3, ans4};
    for (int i = 0; i < 100; i++)
    {
        p->train(dataset, rightAnswers, 0.01);
    }
    for (std::vector<double> data : dataset)
    {
        p->fit(data);
        p->PrintExit();
    }

    p->PrintWeights();

    delete p;
}

void test_multiexit()
{
    std::vector<int> neuronCounts = {2, 3};
    Perceptrone *p = new Perceptrone(neuronCounts, true, RELU);

    std::vector<std::vector<double>> dataset;
    for (int i = 0; i <= 1; i++)
    {
        for (int j = 0; j <= 1; j++)
        {
            std::vector<double> data = {(double)i, (double)j};
            dataset.push_back(data);
        }
    }
    std::vector<double> ans1 = {1, 1, 1};
    std::vector<double> ans2 = {0, 1, 1};
    std::vector<double> ans3 = {1, 0, 1};
    std::vector<double> ans4 = {0, 0, 1};

    std::vector<std::vector<double>> rightAnswers = {ans1, ans2, ans3, ans4};
    for (int i = 0; i < 10000; i++)
    {
        p->train(dataset, rightAnswers, 0.01);
    }

    for (std::vector<double> data : dataset)
    {
        p->fit(data);
        p->PrintExit();
    }

    p->PrintWeights();

    delete p;
}

void testIris()
{
    std::vector<std::vector<double>> train_dataset;
    std::vector<std::vector<double>> train_features;
    std::vector<std::vector<double>> validation_dataset;
    std::vector<std::vector<double>> validation_features;

    std::ifstream train_data("train_dataset.txt");
    if (train_data.is_open())
    {
        int i = 0;

        for (std::string line; train_data >> line;)
        {
            if (i == 0)
            {
                std::vector<double> tmp;
                train_dataset.push_back(tmp);
            }
            if (i < 4)
                train_dataset[train_dataset.size() - 1].push_back(stod(line));
            else
            {
                std::vector<double> feature(3, 0);
                if (line == "Iris-setosa")
                    feature[0] = 1;
                if (line == "Iris-versicolor")
                    feature[1] = 1;
                if (line == "Iris-virginica")
                    feature[2] = 1;
                train_features.push_back(feature);
            }
            i++;
            i%=5;
        }
    }
    train_data.close();

    std::ifstream validation_data("validation_dataset.txt");

    if (validation_data.is_open())
    {
        int i = 0;

        for (std::string line; validation_data >> line;)
        {
            if (i == 0)
            {
                std::vector<double> tmp;
                validation_dataset.push_back(tmp);
            }
            if (i < 4)
                validation_dataset[validation_dataset.size() - 1].push_back(stod(line));
            else
            {
                std::vector<double> feature(3, 0);
                if (line == "Iris-setosa")
                    feature[0] = 1;
                if (line == "Iris-versicolor")
                    feature[1] = 1;
                if (line == "Iris-virginica")
                    feature[2] = 1;
                validation_features.push_back(feature);
            }
            i++;
            i%=5;
        }
    }
    validation_data.close();
    
    std::cout << train_dataset.size() << ' ' << train_features.size() << '\n';
    std::vector<int> neuronCounts = {4, 5, 3};
    Perceptrone* p = new Perceptrone(neuronCounts, true, SIGMOID);
    for (int i = 0; i <= 3000; i++)
    {
        MixDataset(train_dataset, train_features);
        p->train(train_dataset, train_features, 0.001);
    }
    
    std::cout << "Error: " << p->Error(validation_dataset, validation_features) << '\n'; 


    p->PrintWeights();
    p->Save("Iris_nn.txt");
    delete p;
}
int main()
{
    testIris();
    Perceptrone* p = new Perceptrone("Iris_nn.txt");
    std::vector<double> test = {6.3, 3.3, 6.0, 2.5};
    p->fit(test);
    p->PrintExit();
}